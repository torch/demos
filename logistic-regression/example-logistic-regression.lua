--------------------------------------------------------------------------------
-- example-logistic-regression.lua
--
-- Logistic regression and multinomial logistic regression
--

require 'nn'
require 'optim'

-- By setting the random number seed, we get the same results on every run.
torch.manualSeed(123)

--------------------------------------------------------------------------------
-- 1. Create the training data

print('')
print('============================================================')
print('Constructing dataset')
print('')

-- The data come from a tutorial on using R from UCLA, which can be found at
-- http://www.ats.ucla.edu/stat/r/dae/mlogit.htm

-- The model is one of brand preference, where there are 3 brands and 2
-- explanatory variables. The variables are coded this way:
--  brand: 1, 2 or 3
--  female: 1 if the person is a female, 0 if a male
--  age: a positive integer

-- The data are stored in a csv file 'example-logistic-regression.csv'
-- and read with the csvigo package (torch-pkg install csvigo)

require 'csvigo'

-- The data are in a comma separated values (CSV) file. The first record
-- contains field names and subsequent records contain data. The fields and
-- their formats are:
-- - num: observation number; an integer surrounded by double quote chars
-- - brand: brand number; 1, 2, or 3 surrounded by double quote chars
-- - female: indicator for is-female: 1 if female, 0 otherwise; no quote chars
-- - age: age of the person; no quote characters

-- Reading CSV files can be tricky. This code uses the csvigo package for this:
loaded = csvigo.load('example-logistic-regression.csv')

-- We'll end up with lots of data, so lets organize all the data into one
-- table.
data = {}

-- We'll create these fields in the data table:
-- data.raw will hold the data from the CSV file and various related attributes.
-- data.train will hold the training data and various related attributes
-- data.test  will hold the test data and various related attributes

-- Convert the CSV table into dense tensors. The tensor form has the advantage
-- that it stores its elements continguously (which leads to better
-- performance) and a tensor allows one to select columns and rows easily,
-- using slicing methods.

-- First convert each variable list to a tensor and save in an all-purpose data
-- table.
data.raw = {}
data.raw.age = torch.Tensor(loaded.age)
data.raw.brand = torch.Tensor(loaded.brand)
data.raw.isFemale = torch.Tensor(loaded.female)

-- We don't need the loaded data any more. By setting it to nil, the garbage
-- collector will reclaim its storage.

loaded = nil

-- Let's check that we have the same number of samples for each feature.

data.raw.nSamples = data.raw.age:size(1)
assert(data.raw.brand:size(1) == data.raw.nSamples)
assert(data.raw.isFemale:size(1) == data.raw.nSamples)


-- As we debug, we'll want to print the data table in a compact fashion.

require 'printTable'  -- We've put functions that you may find useful in separate files

local function printData()
   printTable('data', data)
end

-- summarize the raw data

require 'summarizeData'
summarizeData(data)

-- The model will use the age and isFemale features to predict brand. The
-- implementation of the model will require the age and isFemale features to be
-- in a tensor, which we call the input in this code. The input tensor is 2D,
-- with one row for each training sample. The predicted feature is called the
-- target, which is held as a 1D tensor.

-- Define the column numbers used for the features in the input 2D tensor.
data.cAge = 1       -- the age feature is always the first element of the input vector
data.cIsFemale = 2  -- the isFemale feature is always the second element of the input vector

local function buildInput(age, isFemale)
   local nSamples = age:size(1)
   local input = torch.Tensor(nSamples, 2)
   for sampleIndex = 1, nSamples do
      input[sampleIndex][data.cAge] = age[sampleIndex]
      input[sampleIndex][data.cIsFemale] = isFemale[sampleIndex]
   end
   return input
end

-- Buid the training data.

data.train = {}
data.train.input = buildInput(data.raw.age, data.raw.isFemale)
data.train.target = data.raw.brand


----------------------------------------------------------------------
-- 2. Define the model (predictor)

-- The model is a multinomial logistic regression. 

-- It will consist of two layers which operate sequentially:
--  - 1: a linear model
--  - 2: a soft max layer

-- The linear model supposes that the un-normalized probability of choosing a
-- specific brand is proportional to the product of unknown weights and the
-- observed variables plus a bias:
--   Prob(brand = b) = bias + weight1 * female * weight2 * age
-- There are two inputs (female and age) and three outputs (one for each value
-- that brand can take on)

linLayer = nn.Linear(2,3)

-- The soft max layer takes the 3 outputs from the linear layer and
-- transforms them to lie in the range (0,1) and to sum to 1. Thus, unlike
-- some text books in which the probabilities are un-normalized, the output
-- of the soft max layer will be normalized probabilities. 

-- The log soft max layer takes the log of these 3 outputs. This is done
-- because we want to feed the log values into the ClassNLLCriterion
-- described below.

softMaxLayer = nn.LogSoftMax()  -- the input and output are a single tensor

-- We need to put the layers into a sequential container.

model = nn.Sequential()
model:add(linLayer)
model:add(softMaxLayer)


----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.

-- In that example, we minimize the cross entropy between the predictions of
-- our linear model and the groundtruth available in the dataset.

-- Torch provides many common criterions to train neural networks.

-- The ClassNLLCriterion expects to be fed the log probabilities in a tensor.
-- Hence, the use of the LogSoftMax layer in the model instead of SoftMax.

-- Minimizing the cross-entropy is equivalent to maximizing the maximum
-- a-posteriori (MAP) prediction, which is equivalent to minimizing the
-- negative log-likelihoood (NLL), hence the use of the NLL loss.

criterion = nn.ClassNLLCriterion()


----------------------------------------------------------------------
-- 4.a. Train the model (Using SGD)

-- To minimize the loss defined above, using the linear model defined in
-- 'model', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function with respect to these
-- parameters by doing so:

x, dl_dx = model:getParameters()

-- The above statement does not create a copy of the parameters in the model!
-- Instead it creates in x and dl_dx a view of the model's weights and
-- derivative with respect to the weights. The view is implemented so that when
-- the weights and their derivatives changes, so do the x and dl_dx. The
-- implementation is efficient in that the underlying storage is shared.

-- A note on terminology: In the machine learning literature, the parameters
-- that one seeks to learn are often called weights and denoted with a W.
-- However, in the optimization literature, the parameter one seeks to 
-- optimize is often called x. Hence the use of x and dl_dx above.

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our mode, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is never made)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > data.raw.nSamples then _nidx_ = 1 end

   local inputs = data.train.input[_nidx_]
   local target = data.train.target[_nidx_]

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using the
-- implementation of stochastic gradient descent in the optim package that
-- comes with torch.
--
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset for a
-- certain number of iterations, and perform a stochastic update at each
-- iteration. The number of iterations is found empirically here, but should
-- typically be determinined using cross-validation (i.e.  using multiple folds
-- of training/test subsets).

epochs = 2e2  -- number of times to cycle over our training data

print('')
print('============================================================')
print('Training with SGD')
print('')

for i = 1, epochs do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   local nSamples = data.raw.nSamples
   for i = 1, nSamples do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific

      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / nSamples
   print(string.format('epoch %5d of %d average loss %f', 
                       i, epochs, current_loss))

end

----------------------------------------------------------------------
-- 4.b. Train the model (Using L-BFGS)

-- Now that we know how to train the model using simple SGD, we can use more
-- complex optimization heuristics. In the following, we use a second-order
-- method: L-BFGS, which typically yields more accurate results (for linear
-- models), but can be significantly slower. For very large datasets, SGD is
-- typically much faster to converge initially and then slows up as it nears a
-- solution. So a common strategy is to start training with SGD and then switch
-- to L-FBGS can be used to refine the results.

-- We start again, and reset the trained parameter vector:

model:reset()

-- Next we re-define the closure that evaluates f and df/dx, so that
-- it estimates the true f, and true (exact) df/dx, over the entire
-- dataset. This is a full batch approach.

feval = function(x_new)
   -- set x to x_new, if differnt
   if x ~= x_new then
      x:copy(x_new)
   end

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- and batch over the whole training dataset:
   local loss_x = 0
   local nSamples = data.raw.nSamples
   for i = 1, nSamples do
      -- select a new training sample
      _nidx_ = (_nidx_ or 0) + 1
      if _nidx_ > nSamples then _nidx_ = 1 end

      local inputs = data.train.input[_nidx_]
      local target = data.train.target[_nidx_]

      -- evaluate the loss function and its derivative wrt x, for that sample
      loss_x = loss_x + criterion:forward(model:forward(inputs), target)
      model:backward(inputs, criterion:backward(model.output, target))
   end

   -- normalize with batch size
   loss_x = loss_x / nSamples
   dl_dx = dl_dx:div( nSamples )

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- L-BFGS parameters are different than SGD:
--   + a line search: we use a line search, which aims at
--                    finding the point that minimizes the loss locally
--   + max nb of iterations: the maximum number of iterations for the batch,
--                           which is equivalent to the number of epochs
--                           on the given batch. In this example, it's simple
--                           because the batch is the full dataset, but in
--                           some cases, the batch can be a small subset
--                           of the full dataset, in which case maxIter
--                           becomes a more subtle parameter.

lbfgs_params = {
   lineSearch = optim.lswolfe,
   maxIter = epochs,
   verbose = true
}

print('')
print('============================================================')
print('Training with L-BFGS')
print('')

_,fs = optim.lbfgs(feval,x,lbfgs_params)

-- fs contains all the evaluations of f, during optimization

print('history of L-BFGS evaluations:')
for i = 1, #fs do
   print(i, fs[i])
end

----------------------------------------------------------------------
-- 5. Refactor the code
--
-- The two training schemes, one for SGD and one for L-BFGS, work, but there is
-- redundant code and the code is not packaged in a way that would let one move
-- it easily to another program without copying and splicing. Some of the code,
-- like the model:reset() call is tricky.  Let's fix that.

-- The first thing to fix is to encapsulate the definition of logistic
-- regression into a single function that returns the model and criterion. In
-- this example, the criterion is not regularized, but in your real-world
-- applications, you will often want to regularize the model.

require 'logregModelCriterion'

-- Packaging the fitting procedures as external functions is also desirable.
-- The fitting procedure returns a trained model that depends on the training
-- data (the inputs and targets) and number of classes, the algorithm (in this
-- case SGD or L-BFGS), the stopping criteria for the algorithm (we only used
-- the number of epochs in this example, but other criteria are possible and
-- often used), and how the training data are sampled. The fitting procedure
-- results in a fitted model (which has the parameters set) and some additional
-- information about the fitting procedure.

require 'logregFit'

      
---------------------------------------------------------------------
-- 6. Scaling the data
--
-- We have one more detail to take care of. The input variables for the model
-- are the indicator variable for if the subject was female and the age of the
-- subject. The range of the is-female variable is [0, 1] with an average of
-- 0.5. The range of the age variable is [24, 38] with an average of about 31.
-- The difference in magnitude will make the fitting procedure for SGD
-- difficult: along one axis of the objective function, a very small step size
-- will be needed and a larger step size will be usable along the other axis.
-- We must run SGD with the smaller of these step sizes or SGD will not
-- converge. Using a very small step size will cause very slow convergence
-- along the axis that would permit a large step size.
--
-- You may have noticed that we didn't show how accurate the SGD model was in
-- predicting the brand based on age and is_female. That's because its terrible
-- using the sgd_params we set. 
--
-- The fix is to transform all the input variables so that they are on a
-- similar scale. One way to do that is to determine the mean of each variable
-- and its standard deviation. The variable x is then replaced by (x - mean) /
-- standard_deviation.
--
-- By the way, you should always put the training features of your model on
-- a similar scale. 
--
-- We'll need to keep track of the mean and standard deviation in order to make
-- predictions with new data. This function takes care of that and carries out
-- the standardization transformation.

require 'standardize'

-- Let's scale the training data. The unscaled data are in data.raw. We need to
-- save the means and standard deviations for use when we are testing.

local ageScaled, mean, stdv = standardize(data.raw.age)
data.train.ageScaled = ageScaled
data.train.ageMean = mean
data.train.ageStandardDeviation = stdv

-- You don't need to scale the isFemale feature, because its either 0 or 1.  We
-- scale it for consistency in the code.

local isFemaleScaled, mean, stdv = standardize(data.raw.isFemale)
data.train.isFemaleScaled = isFemaleScaled
data.train.isFemaleMean = mean
data.train.isFemaleStandardDeviation = stdv

data.train.inputScaled = 
   buildInput(data.train.ageScaled, data.train.isFemaleScaled)



-------------------------------------------------------------------------
-- 7. Training the models on the scaled data
--
-- We train first using SGD and then using L-BFGS. 

local nClasses = 3

-- Fit using SGD and one sample at a time in order of the training data.  It
-- would be better to randomize the order of the training data, because the
-- training samples may be ordered in a way this is disadvantageous to the
-- training procedure. For example, the training samples may be ordered so that
-- similar samples are grouped together. If you are interested in randomizing
-- the training order, you would modify the code in the module logregFitSgd.

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

sgd_stopping_criteria = {max_epochs = 200}

modelFittedSgd, sgdFittingInfo = 
   logregFit('SGD', sgd_params, 'sequential-1', sgd_stopping_criteria, 
             data.train.inputScaled, data.train.target, nClasses)

assert(sgdFittingInfo.reason_stopped == 'max_epochs') 

-- Ffit using L-BFGS, approximating the Hessian using all the training vectors.
-- Thus the algorithm is BFGS. This works here because there are a few
-- variables.  If you have more variables, you will want to fit L-BFGS using a
-- mini batch. Some recommend a mini-batch of about 1,000 samples.

local epochs = 200
lbfgs_params = {
   lineSearch = optim.lswolfe,
   maxIter = epochs,
   verbose = true
}

lbfgs_stopping_criteria = {}

modelFittedLbfgs, LbfgsFittingInfo = 
   logregFit('L-BFGS', lbfgs_params, 'entire-batch', lbfgs_stopping_criteria, 
             data.train.inputScaled, data.train.target, nClasses)

assert(LbfgsFittingInfo.reason_stopped == 'lbfgsParams')

-- print the losses at each step of the L-BFGS fitting
local losses = LbfgsFittingInfo.losses
for i = 1, #losses do
   print('lbfgs step ' .. i .. ' loss ' .. losses[i])
end

-------------------------------------------------------------------------
-- 8. Test the trained models.
--
-- The testing code is in a separate module, so as to not clutter the logic
-- flow.  Function test_model takes a fitted model and all the data including
-- test data and returns the error rate on the test data. As a side effect, the
-- testing code prints a continency matrix and compares the results on the test
-- data with the results from the text.

require 'testModel'

-- We generate one test example for each possible value of age and isFemale.
--
-- Because we have standardized the training data, we need to standardize the
-- test data as well. To do that we pass in the means and standard deviations
-- from the standardization.

local function makeExample(age, isFemale)
   local input = torch.Tensor(2)
   input[data.cAge] = standardize(torch.Tensor{age}, 
                                  data.train.ageMean, 
                                  data.train.ageStandardDeviation)
   input[data.cIsFemale] = standardize(torch.Tensor{isFemale}, 
                                       data.train.isFemaleMean, 
                                       data.train.isFemaleStandardDeviation)
   return input
end

local nTestExamples = data.raw.nAge * data.raw.nIsFemale
local nFeatures = 2
local inputs = torch.Tensor(nTestExamples, nFeatures)

local nextInputIndex = 0
for age = data.raw.ageMin, data.raw.ageMax do
   for isFemale = data.raw.isFemaleMin, data.raw.isFemaleMax do
      nextInputIndex = nextInputIndex + 1
      inputs[nextInputIndex] = makeExample(age, isFemale)
   end
end
data.test = {}
data.test.input = inputs

-- Determine the accuracy of our models vs. the model from the text.

errorRateSgd = testModel(modelFittedSgd, data, 'SGD', sgd_params)
assert(errorRateSgd == 0)

errorRateLbfgs = testModel(modelFittedLbfgs, data, 'L-BFGS', lbfgs_params)
assert(errorRateLbfgs == 0)

print('done') 
