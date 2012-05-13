-- example-logistic-regression-using-class-LogisticRegression.lua
-- Multinomial Logistic Regression using the LogisticRegression class

-- This is a reworking of example-logistic-regression.lua:
-- + same data set (from UCLA)
-- + instead of writing out the model and optimization code, this version
--   uses the LogisticRegression class that does that for you.
-- + comments in this code are abbreviated, because most of the details
--   are explained in example-logistic-regression.lua

-- from torch 
require 'nn'

-- from examples directory
require 'LogisticRegression'

--------------------------------------------------------------------------------
-- 1. Create the training data

-- The training data are kept in two parallel arrays.
features = {} -- each element will contain a Tensor holding age, female
targets = {}  -- each element will contain a number holding the brand number

do
   -- Directly parsing a Csv file is complex because of the rules
   -- for quoting fields that contain spaces or quote characters. This 
   -- logic is implemented in the Csv class.

   -- If you know your csv file does not use quoting or uses only simple
   -- quoting, you can read it about 3 times faster by writing a loop 
   -- and using string.match to read the file.

   -- The code below shows how to do this. Note that this csv file uses
   -- "simple quoting" defined as no quote character is every escaped within
   -- a quoted field.

   local csvFile = io.open('example-logistic-regression.csv', 'r')
   -- read and check the header
   local header = csvFile:read()
   print('header', header)
   assert(header == '"num","brand","female","age"', 
          'header not as expected')
   -- read and save each data line
   for line in csvFile:lines('*l') do
      local num, brand, female, age = 
         string.match(line, '^"(%d+)","(%d+)",(%d+),(%d+)')
      -- the "+ 0" arithmetic converts the strings to numbers
      -- store the data in a two parallel arrays
      features[#features + 1] = torch.Tensor({age + 0, female + 0}) --a tensor
      targets[#targets + 1] = brand + 0                        -- a lua number
   end
   csvFile:close()
end


-- print the first few rows from the file
print("brand age female")
for i=1,10 do
   print(string.format('%5d %3d %6d', 
                       targets[i], features[i][1], features[i][2]))
end


--------------------------------------------------------------------------------
-- 2. Define how to access the training data

--[[

The LogisticRegression class provides a flexible interface to your
training data in the hope that you do not have to re-organize it to
use the LogisticRegression class.

The training samples are in two groups: the features and the targets.
Naturally, for every feature there must be a target, which is a class
designator. The class designator can be any torch value. Often, as in
this example, it is an integer, the number of the class.


The targets[i] must yield a number which is the class label. Any object
can be passed, provided that either

++ targets is a table, in which case the value at key i is returned or 
   if it is missing, the value of the index metamethod of targets is
   returns
++ targets is not a table, in which case the value returned by the 
   metamethod index is used

Here our targets variables is an array, so its a table, and simple
indexing will work. So we can pass the targets variable directly.

--]]

-- define the LogisticRegression object
local numDimensions = 2
local numClasses = 3
logisticRegression = LogisticRegression(features, targets,
                                        numClasses, numDimensions)


--------------------------------------------------------------------------------
-- 3. Train the model with stochastic gradient descent

print('')
print('============================================================')
print('Training with SGD')

--[[

To train the model, we need to:

a. Define the parameters used to control the training procedure.

b. Tell the training procedure how to interpret the features and targets
   variables passed during contruction.

--]]

-- 3a. Defining the parameters.

opt = {
   algo = 'sgd',              -- use stochastic gradient descent
   numEpochs = 100,           -- for 100 epochs
   validate = true,           -- check the opt table for correctness
   verboseBatch = false,      -- don't print results for each batch
   verboseEpoch = true,       -- do print results for each epoch
   optimParams = {learningRate = 1e-3,       -- initial learning rate
                  learningRateDecay = 1e-4}, -- how initial rate decays
}

function printOptimizationParameters(opt)
   print()
   print('optimization parameters')
   for k,v in pairs(opt) do
      print(k,v)
   end
end

printOptimizationParameters(opt)

--[[ 3b. Define how to find the training samples.

We have defined how to do the optimization. Now we need to define how
the training procedure can find our samples. Each sample is made up of
a feature (which must be a 1D tensor) and a target (which must be a
Lua number). When we constructed logisticRegression, we passed
features and targets as parameters. Now we tell the trainer how to
interpret those variables.

Some of the optimization procedures work on just one sample at a time.
Others can work on multiple samples at at time. The sequence of
samples that the optimizatio procedure works on is called a "batch."
This term is used for a one-sample batch and for a multi-sample
batch. Another term used is "mini-batch", typically when the sequence
of samples presented to the algorithm is small fraction of all the
training samples.

The description of how to interpret the features and targets variables
is always in terms of batches.

A batch is composed of indices which may be numeric or more general
keys.  A sample is accessed by the optimization routine by accessing
"features[i]" and "targets[i]", where "i" is a key in a batch.

You can put your features and targets into any object that make sense
for the rest of your program provided that you write a function
"nextBatch" that iterates over the features and targets.

The nextBatch function is modeled on the Lua built-in function next,
which is used to iterate over table. The "next(table, key)" function
has this API:

+ table   : a Lua table
+ key     : a key to the table
+ result1 : if key == nil then
              result1 is an initial index into the table
              result2 is the value in the table at index result1
            else if key is the last index in the table then
              result1 is nil
              result2 is ignored
            else
              result1 is the next index in the table
              result2 is the value in the table at index result1
+ result2 : if result1 == nil then
               not provided
            else
               the value table[key]

The Lua next function can be used to iterate over a Lua table like this

  for k,v in next, table do <use the key k and element v> end

The API for nextBatch is similar to that of Lua's next function, except that
the keys are tables of indices and only one value is returned. The
"nextBatch(features, keys)" function has this API:

+ features : any Torch object (often a table or 2D tensor). This is 
             the same features object passed to the Trainer's constructor.
+ keys     : either nil or a table of indices or keys for features.
+ result   : nil, if keys is the last set of indices for features in the batch;
           : a table containing the indices in the next batch, if keys
           :   is not the last set of indices for features in the batch

The features and targets are retrieved by the Trainer through this iteration:

   local batchIndices = nextBatch(features, nil)
   while batchIndices do
      for _,index in ipairs(batchIndices) do
          local feature = features[index]
          local target = targets[index]
          <learn using feature (a 1D tensor)  and target (a number)>
      end -- iteration over elements of the batch
      batchIndices = nextBatch(features, batchIndices)                 
   end -- iteration over batches               

--]]

-- Define nextBatch iterator for the features and targets which
-- are stored in parallel arrays. The batch size is 1, so that each training
-- sample is presented to the optimization procedure one by one.
function nextBatchSgd(features, prevIndices)
   if prevIndices == nil then
      return {1}
   else
      assert(#prevIndices == 1)
      local onlyIndex = prevIndices[1]
      if onlyIndex >= #features then
         return nil
      else
         return {onlyIndex + 1}  -- return table with one element
      end
   end
end

-- Train the model. The printout will contain the loss function values
logisticRegression:train(nextBatchSgd, opt)

--------------------------------------------------------------------------------
-- 4. Continue training with L-BFGS pickup up where SGD left off

print('')
print('============================================================')
print('Restart Training with L-BFGS')
print()

--[[

L-BFGS is designed as a batch optimization method, so it expects more
than one sample of the data to be iterated over. The Wikipedia article
on L-BFGS explains why more than one sample is needed.

--]]

-- Define a nextBatch iterator for the features and targets which
-- are stored in parallel arrays. The batch size is the entire set of
-- samples.
function nextBatchLbfgs(features, prevIndices)
   if prevIndices == nil then
      -- return all the indices
      local nextIndices = {}
      for i,_ in ipairs(features) do
         nextIndices[#nextIndices + 1] = i
      end
      return nextIndices
   else
      return nil -- there is only one batch
   end
end

-- define the options L-BFGS
opt = {
   algo = 'lbfgs',                -- use limited memory BFGS
   numEpochs = 2,               -- for 100 epochs STOP EARLY FOR DEBUGGING
   validate=true,                 -- check the opt table for correctness
   verboseBatch = true,           -- print results for each batch ...
   verboseEpochs = true,          -- .. and for each epoch
   algoParms= {validate=true},    -- 
   optimParams = {
      lineSearch = optim.lswolfe, -- line search satisfying Wolfe conditions
                 --tolFun = 1e-10,  -- default is 1e-5
                   maxIter=100,    -- up to 100 iterations
                   verbose = true} -- print info as we go
}

printOptimizationParameters(opt)

-- Continue training where we left off.
logisticRegression:train(nextBatchLbfgs, opt)

-- CLEMENT: IS THIS MEANT TO BE TRUE?
-- If you run this program several times, you will notice that 
-- sometimes the final loss for L-BFGS is less than for SGD and
-- sometimes its not. This is because L-BFGS uses the last vectors
-- to approximate the true Hessian. Thus the method is sensitive 
-- to the starting point, which is where the SGD method left off.

halt()

--------------------------------------------------------------------------------
-- 5. Train using only L-BFGS

print('')
print('============================================================')
print('Restart Training with L-BFGS')
print()

newLogisticRegression = LogisticRegression(features, pairsFeatures, targets)
-- define the parameters for L-BFGS
opt = {
   algo = 'lbfgs',
   algoParms={validate=true},
   optimParms = {lineSearch = optim.lswolfe,
                 verbose = true,
                 maxIter = 100}
}

-- train from a random starting point
newLogisticRegression:train(opt)

print('finished')

--------------------------------------------------------------------------------
-- 6. Train with L-BFGS in mini batch of 10 samples

print('============================================================')
print('Restart Training with L-BFGS and mini-batches')
print()

print("WRITE ME")

--------------------------------------------------------------------------------
-- 7. Show how to put the features and targets into one object

--[[

So far, the features and targets have each been in their own variables
and these variables have been parallel arrays.

This section demonstrates how to put the features and targets into a
single data structure.

We start by reading in the data again.

--]]

samples = torch.tensor(735, 3)

-- samples[i][1] is the target value (the brand) for sample i
-- samples[i][2] is the value of female for sample i
-- samples[i][3] is the value of age for sample i
indexBrand = 1
indexFemale = 2
indexAge = 3

do
   local csvFile = io.open('example-logistic-regression.csv', 'r')
   -- read and check the header
   local header = csvFile:read()
   print('header', header)
   assert(header == '"num","brand","female","age"', 
          'header not as expected')

   function safeToNumber(s)
      local result = tonumber(s)
      if result then return result end
      error(s .. ' is not convertable to a number')
   end
   
   -- read and save each data line
   local sampleNumber = 0
   for line in csvFile:lines('*l') do
      sampleNumber = sampleNumber + 1
      local num, brand, female, age = 
         string.match(line, '^"(%d+)","(%d+)",(%d+),(%d+)')
      samples[sampleNumber][indexBrand] = safeToNumber(brand)
      samples[sampleNumber][indexFemale] = safeToNumber(female)
      samples[sampleNumber][indexAge] = safeToNumber(age)
   end
   csvFile:close()
end

-- Define nextBatch iterator to use samples. The batch size is 1.
-- This is exactly the same function used for the Stochastic Gradient
-- Descent example.
function nextBatchOneObject(features, prevIndices)
   if prevIndices == nil then
      return {1}
   else
      assert(#prevIndices == 1)
      local onlyIndex = prevIndices[1]
      if onlyIndex >= #features then
         return nil
      else
         return {onlyIndex + 1}  -- return table with one element
      end
   end
end

-- The Trainer class requires that features[i] return a tensor
-- and samples[i] return a scalar. We define two object that
-- do just that

features = {}
targets = {}

-- Return the tensor which is the feature vector for sample key.
-- The first argument is not used.
function getFeatures(table, key)
   assert(type(key) == 'number', key ..  ' is not a number')
   assert(key <= i and key <= 735, key .. ' is not in range')
   -- CLEMENT: HOW TO DEFINE A VIEW INSTEAD OF A NEW TENSOR?
   local result = torch.Tensor(2)
   result[1] = samples[key][indexFemale]
   result[2] = samples[key][indexAge]
   return result
end

-- Return a number which is the brand identifier for sample key.
-- The first argument is not used.
function getTarget(table, key)
   assert(type(key) == 'number', key ..  ' is not a number')
   assert(key <= i and key <= 735, key .. ' is not in range')
   return samples[key][indexBrand]
end

setmetatable(features, {__index == getFeatures})
setmetatable(targets, {__index = getTarget})

-- setup the optimization parameters
opt = {
   algo = 'sgd',              -- use stochastic gradient descent
   numEpochs = 100,           -- for 100 epochs
   validate = true,           -- check the opt table for correctness
   verboseBatch = false,      -- don't print results for each batch
   verboseEpoch = true,       -- do print results for each epoch
   optimParams = {learningRate = 1e-3,       -- initial learning rate
                  learningRateDecay = 1e-4}, -- how initial rate decays
}

-- Train the model. The printout will contain the loss function values
logisticRegression:train(nextBatchOneObject, opt)

print()
print('Finished')



