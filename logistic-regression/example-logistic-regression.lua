----------------------------------------------------------------------
-- example-logistic-regression.lua
--
-- Logistic regression and multinomial logistic regression
--

require 'nn'
require 'optim'


----------------------------------------------------------------------
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
-- and read through the class Csv (required from ./csv.lua)

require 'csv'

-- Build a table with one row for each observation
-- Each row will contain {brand, female, age}
rows = {}

-- The data are in a comma separated values (CSV) file. The first record
-- contains field names and subsequent records contain data. The fields and
-- their formats are:
-- - num: observation number; an integer surrounded by double quote chars
-- - brand: brand number; 1, 2, or 3 surrounded by double quote chars
-- - female: indicator for is-female: 1 if female, 0 otherwise; no quote chars
-- - age: age of the person; no quote characters

-- Reading CSV files can be tricky. This code uses the Csv class for this.

-- establish a scope to keep the global name space clean
do 
   local csv = Csv('example-logistic-regression.csv', 'r') -- r means read mode
   -- read and check the header
   local header = csv:read()  -- header is an array of strings
   assert(header[1] == 'num', 'num not found where expected')
   assert(header[2] == 'brand', 'brand not found where expected')
   assert(header[3] == 'female', 'female not found where expected')
   assert(header[4] == 'age', 'age not found where expected')
   -- read and save each data line
   while true do
      local dataLine = csv:read()
      -- dataLine is null if we have reached the end of file
      if not dataLine then break end
      -- the values are strings, so convert them to numbers by adding 0
      rows[#rows+1] = {dataLine[2] + 0,
		       dataLine[3] + 0,
		       dataLine[4] + 0}
   end
   csv:close()
end

-- print the first few rows
print("brand female age")
for i=1,10 do
   print(string.format('%5d %6d %3d', rows[i][1], rows[i][2], rows[i][3]))
end

-- Convert the rows table into a 2D Torch Tensor. The tensor form has the
-- advantage that it stores its elements continguously (which leads to
-- better performance) and a tensor allows one to select columns and rows
-- easily, using slicing methods.

data = torch.Tensor(rows)

brands  = data[{ {}, {1} }]  -- the entire first column
females = data[{ {}, {2} }]  -- the entire second column
ages    = data[{ {}, {3} }]  -- the entire third column

-- To implement the model, we need to know how many categories there are.
numberOfBrands = 0
do
   seen = {}
   for i = 1,brands:size(1) do
      -- brands[i] yields a 1D tensor
      local nextBrand = brands[i][1]  -- extract the integer value
      if not seen[nextBrand] then 
         seen[nextBrand] = true
      end
   end
   numberOfBrands = #seen
end

-- summarize the data
function summarizeData()
   function p(name,value) 
      print(string.format('%20s %f', name, value) )
   end
   p('number of brands', numberOfBrands)
   p('min brand', torch.min(brands))
   p('max brand', torch.max(brands))
   
   p('min female', torch.min(females))
   p('max female', torch.max(females))
   
   p('min age', torch.min(ages))
   p('max age', torch.max(ages))
end

summarizeData()

-- check that the number of brands is exactly equal to the max brand value
if torch.max(brands) ~= numberOfBrands then
   error('number of brands is off')
end


----------------------------------------------------------------------
-- 2. Define the model (predictor)

-- The model is a multinomial logistic regression. 

-- It will consist of two layers that operate sequentially:
--  - 1: a linear model
--  - 2: a soft max layer

-- The linear model supposes that the un-normalized probability of choosing
-- a specific brand is proportional to the product of unknown weights and 
-- the observed variables plus a bias:
--   Prob(brand = b) = bias + weight1 * female * weight2 * age
-- There are two inputs (female and age) and three outputs (one for each
-- value that brand can take on)

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

-- In that example, we minimize the cross entropy between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.

-- Torch provides many common criterions to train neural networks.

-- The ClassNLLCriterion expects to be fed the log probabilities in a
-- tensor. Hence, the use of the LogSoftMax layer in the model instead
-- of SoftMax.

-- Minimizing the cross-entropy is equivalent to maximizing the 
-- maximum a-posteriori (MAP) prediction, which is equivalent to 
-- minimizing the negative log-likelihoood (NLL), hence the use of
-- the NLL loss.

criterion = nn.ClassNLLCriterion()


----------------------------------------------------------------------
-- 4.a. Train the model (Using SGD)

-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these 
-- parameters by doing so:

x, dl_dx = model:getParameters()

-- The above statement does not create a copy of the parameters in the 
-- model! Instead it create in x and dl_dx a view of the model's weights
-- and derivative wrt the weights. The view is implemented so that when
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
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ 1 }]        -- this funny looking syntax allows
   local inputs = sample[{ {2,3} }]    -- slicing of arrays.

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
   
   -- report on the parameters, which should be converging
   if false then
      local p = model:getParameters()
      print(string.format(
	    "parameters %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f", 
	    p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
	   )
   end

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
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

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation (i.e.
-- using multiple folds of training/test subsets).

epochs = 1e2  -- number of times to cycle over our training data

print('')
print('============================================================')
print('Training with SGD')
print('')

for i = 1,epochs do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#data)[1] do

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
   current_loss = current_loss / (#data)[1]
   print('epoch = ' .. i .. 
	 ' of ' .. epochs .. 
	 ' current loss = ' .. current_loss)

end


----------------------------------------------------------------------
-- 4.b. Train the model (Using L-BFGS)

-- now that we know how to train the model using simple SGD, we can
-- use more complex optimization heuristics. In the following, we
-- use a second-order method: L-BFGS, which typically yields
-- more accurate results (for linear models), but can be significantly
-- slower. For very large datasets, SGD is typically much faster
-- to converge, and L-FBGS can be used to refine the results.

-- we start again, and reset the trained parameter vector:

model:reset()

-- next we re-define the closure that evaluates f and df/dx, so that
-- it estimates the true f, and true (exact) df/dx, over the entire
-- dataset. This is a full batch approach.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- an batch over the whole training dataset:
   local loss_x = 0
   for i = 1,(#data)[1] do
      -- select a new training sample
      _nidx_ = (_nidx_ or 0) + 1
      if _nidx_ > (#data)[1] then _nidx_ = 1 end

      local sample = data[_nidx_]
      local target = sample[{ 1 }]        -- this funny looking syntax allows
      local inputs = sample[{ {2,3} }]    -- slicing of arrays.

      -- evaluate the loss function and its derivative wrt x, for that sample
      loss_x = loss_x + criterion:forward(model:forward(inputs), target)
      model:backward(inputs, criterion:backward(model.output, target))
   end

   -- normalize with batch size
   loss_x = loss_x / (#data)[1]
   dl_dx = dl_dx:div( (#data)[1] )

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- L-BFGS parameters are different than SGD:
--   + a line search: we provide a line search, which aims at
--                    finding the point that minimizes the loss locally
--   + max nb of iterations: the maximum number of iterations for the batch,
--                           which is equivalent to the number of epochs
--                           on the given batch. In that example, it's simple
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
print(fs)


----------------------------------------------------------------------
-- 5. Test the trained model.

print('')
print('============================================================')
print('Testing the model')
print('')

-- Now that the model is trained, one can test it by evaluating it
-- on new samples.

-- The model constructed and trained above computes the probabilities
-- of each class given the input values.

-- We want to compare our model's results with those from the text.
-- The input variables have narrow ranges, so we just compare all possible
-- input variables in the training data.

-- Determine actual frequency of the each female-age pair in the 
-- training data

-- return index of largest value
function maxIndex(a,b,c)
   if a >=b and a >= c then return 1 
   elseif b >= a and b >= c then return 2
   else return 3 end
end

-- return predicted brand and probabilities of each brand
-- for the model in the text

-- The R code in the text computes the probabilities of choosing
-- brands 2 and 3 relative to the probability of choosing brand 1:
--   Prob(brand=2)/prob(brand=1) = exp(-11.77 + 0.52*female + 0.37*age)
--   Prob(brand=3)/prob(brand=1) = exp(-22.72 + 0.47*female + 0.69*age)
function predictText(age, female)
   --   1: calculate the "logit's"
   --      The coefficients come from the text.
   --      If you download the R script and run it, you may see slightly
   --      different results.
   local logit1 = 0
   local logit2 = -11.774655 + 0.523814 * female + 0.368206 * age
   local logit3 = -22.721396 + 0.465941 * female + 0.685908 * age

   --   2: calculate the unnormalized probabilities
   local uprob1 = math.exp(logit1)
   local uprob2 = math.exp(logit2)
   local uprob3 = math.exp(logit3)

   --   3: normalize the probabilities
   local z = uprob1 + uprob2 + uprob3
   local prob1 = (1/z) * uprob1
   local prob2 = (1/z) * uprob2
   local prob3 = (1/z) * uprob3

   return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
end

-- return predicted brand and the probabilities of each brand
-- for our model
function predictOur(age, female)
   local input = torch.Tensor(2)
   input[1] = female  -- must be in same order as when the model was trained!
   input[2] = age
   local logProbs = model:forward(input)  
   --print('predictOur', age, female, input)
   local probs = torch.exp(logProbs)
   --print('logProbs', logProbs)
   --print('probs', probs[1], probs[2], probs[3] )
   local prob1, prob2, prob3 = probs[1], probs[2], probs[3]
   return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
end
      
counts = {}

function makeKey(age, brand, female)
   -- return a string containing the values

   -- Note that returning a table will not work, because each
   -- table is unique.

   -- Because Lua interns the strings, a string with a given sequence
   -- of characters is stored only once.
   return string.format('%2d%1d%1f', age, brand, female)
end

for _,row in pairs(rows) do
   local brand = row[1]
   local female = row[2]
   local age = row[3]
   local key = makeKey (age, brand, female)
   counts[key] = (counts[key] or 0) + 1
end

-- return probability of each brand conditioned on age and female
function actualProbabilities(age, female)
   function countOf(age, brand, female)
      return counts[makeKey(age, brand, female)] or 0
   end
   local count1 = countOf(age, 1, female)
   local count2 = countOf(age, 2, female)
   local count3 = countOf(age, 3, female)
   local sumCounts = count1 + count2 + count3
   if sumCounts == 0 then
      return 0, 0, 0
   else
      return count1/sumCounts, count2/sumCounts, count3/sumCounts
   end
end


print(' ')
print('summary of data')
summarizeData()

print(' ')
print('training variables')
for k,v in pairs(sgd_params) do
   print(string.format('%20s %f', k, v))
end
print(string.format('%20s %f', 'epochs', epochs))

print(' ')
print('current loss', current_loss)

-- print the headers 
print(' ')
lineFormat = '%-6s %-3s| %-17s | %-17s | %-17s | %-1s %-1s %-1s'
print(
   string.format(lineFormat,
		 '', '', 
		 'actual probs', 'text probs', 'our probs', 
		 'best', '', ''))
choices = 'brnd1 brnd2 brnd3'
print(string.format(lineFormat,
		    'female', 'age', 
		    choices, choices, choices, 
		    'a', 't', 'o'))

-- print each row in the table

function formatFemale(female)
   return string.format('%1d', female)
end

function formatAge(age)
   return string.format('%2d', age)
end

function formatProbs(p1, p2, p3)
   return string.format('%5.3f %5.3f %5.3f', p1, p2, p3)
end

function indexString(p1, p2, p3)
   -- return index of highest probability or '-' if nearly all zeroes
   if p1 < 0.001 and p2 < 0.001 and p3 < 0.001 then
      return '-'
   else 
      return string.format('%1d', maxIndex(p1, p2, p3))
   end
end

-- print table rows and accumulate accuracy
for female = 0,1 do
   for age = torch.min(ages),torch.max(ages) do
      -- calculate the actual probabilities in the training data
      local actual1, actual2, actual3 = actualProbabilities(age, female)
      -- calculate the prediction and probabilities using the model in the text
      local textBrand, textProb1, textProb2, textProb3 = 
	 predictText(age, female)
      -- calculate the probabilities using the model we just trained
      --print("main", age, female)
      local ourBrand, ourProb1, ourProb2, ourProb3 = 
	 predictOur(age, female)
      print(
	 string.format(lineFormat,
		       formatFemale(female), 
		       formatAge(age),
		       formatProbs(actual1, actual2, actual3),
		       formatProbs(textProb1, textProb2, textProb3),
		       formatProbs(ourProb1, ourProb2, ourProb3),
		       indexString(actual1,actual2,actual3),
		       indexString(textProb1,textProb2,textProb3),
		       indexString(ourProb1,ourProb2,ourProb3))
	   )
   end
end


----------------------------------------------------------------------
-- 6. Assess accuracy on the training data.

-- We can compare our model with the UCLA model.

-- The table generated just above shows the brand that each model would
-- have predicted if it were to predict the brand that it estimated had
-- the highest probability. It also shows the distribution of actual
-- brand choices. All these results are conditioned on a specific value
-- for the female and age variables. The predictions are in the last
-- 3 columns of the table, where 'a' means the value predicted from the 
-- actual data, 't' means the value predicted from the text, and 'o' means
-- the value predicted from our model.

-- One can see that when our estimate differs from that implied by the 
-- actual data, our predicted probabilities are very different from the
-- actual probabilities. Also on (female,age) pairings where our model
-- differs from the actual, the text model is often correct.

-- Both the text model and our model use the same cost function and the
-- same parameters. The text model finds the parameters by using BFGS 
-- whereas our model uses stochastic gradient descent.

print('')
print('============================================================')
print('Accuracy on training data')
print('')

countOfDifferences = {}
function makeKey(female, age, actualBrand, textPrediction, ourPrediction)
   return string.format('%6d %3d %6d %3d %4d',
			female, age, 
			actualBrand, textPrediction, ourPrediction)
end

for _,row in pairs(rows) do
   local actualBrand = row[1]
   local female = row[2]
   local age = row[3]
   local textPrediction = predictText(age, female)
   local ourPrediction = predictOur(age, female)
   if actualBrand == textPrediction then
      numberCorrect = (numberCorrect or 0) + 1
      countTextCorrect = (countTextCorrect or 0) + 1
   end
   if actualBrand == ourPrediction then
      numberCorrect = (numberCorrect or 0) + 1
      countOurCorrect = (countOurCorrect or 0) + 1
   end
   if (numberCorrect ~= 2) and (textPrediction ~= ourPrediction) then
      local key = makeKey(female,age,actualBrand,textPrediction,ourPrediction)
      countOfDifferences[key] = (countOfDifferences[key] or 0) + 1
   end
   countObservations = (countObservations or 0) + 1
end

print()
print("Differences between text's predictions and our predictions")
print("When only one prediction is correct")
print('female age actual text our occurs')
accuracyLine = '%-26s %6d'
table.sort(countOfDifferences)
for k,v in pairs(countOfDifferences) do
   print(string.format('%-26s %6d', k, v))
end

print()
function printCorrectFraction(name, numCorrect)
   print(
      string.format(
	 'Fraction of %s predictions that were correct on training set = %f',
	 name, numCorrect / countObservations))
end

printCorrectFraction('text', countTextCorrect)
printCorrectFraction('our', countOurCorrect)

print(' ')
print('Frequency of each brand in training data')

function countBrand(brandNumber)
   local count = 0
   for _,row in pairs(rows) do
      local actualBrand = row[1]
      if actualBrand == brandNumber then
	 count = count + 1
      end
   end
   return count
end

function printFrequency(brandNumber)
   print(string.format('Brand %d occurs %f',
		       brandNumber, countBrand(brandNumber) / #rows))
end

printFrequency(1)
printFrequency(2)
printFrequency(3)


-- Note that our predictions are less accurate than simply predicting
-- brand 2 no matter what age and female are!

