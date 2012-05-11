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
require 'logisticregression'

--------------------------------------------------------------------------------
-- 1. Create the training data

-- The training data are kept in two arrays
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
   -- "simple quoting" defined ast no quote character is every escaped within
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
      -- store the data in an array (rows) where each element is a 1D tensor
      features[#features + 1] = torch.Tensor({age + 0, female + 0})
      targets[#targets + 1] = brand + 0
   end
   csvFile:close()
end


-- print the first few rows from the file
print("brand age female")
for i=1,10 do
   print(string.format('%5d %3d %6d', 
                       targets[i], features[i][1], features[i][2]))
end

-- WOULD BE NICE TO ELIMINATE THIS CONVERSION
if false then
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

end

--------------------------------------------------------------------------------
-- 2. Define how to access the training data

-- The LogisticRegression class provides a flexible interface to your training
-- data in the hope that you do not have to re-organize it to use the 
-- LogisticRegression class.

-- The training data are in two groups: the features and the targets. 
-- Naturally, for every feature there must be a target, which is a class
-- designator. The class designator can be any torch value. Often it is 
-- an integer, the number of the class.

-- You can put your features into any object you desire in any form that
-- make sense for the rest of your program provided that you can write
-- a function "pairsFeatures(features)" that returns an iterator over the
-- features. The iterator must return either a pair (index, tensor) or the 
-- single value nil on each call.  

-- The nil value is a sentinel to signal that all features have been presented. 

-- The (i, tensor) pair when returned means that there is another training
-- sample, namely the tensor. It must be a 1D Tensor.

-- The i value can be any value at all. The class corresponding to the
-- tensor must be the value returned by targets[i]. Most simply, i can be
-- an integer and targets an array or tensor.

-- The pairsFeatures function has exactly the same interface as 
-- the built-in Lua function ipairs(array).

-- We need an iterator function that takes the features data structure and 
-- an index and returns the next index and feature or nil. 

function iteratorFeatures(features, index)
   index = index + 1
   -- because features is an array, the test for the presence of the
   -- next feature is simple
   if index > #features then
      return nil
   else
      return index, features[index]
   end
end

-- The pairsFeatures function takes a features object and returns the
-- the values expected by the Lua for statements:
-- + an iterator function, in this case, iteratorFeatures
-- + an invariant, the features themselves. The invariant is the first
--   argument to the iterator function
-- + the initial index value, which is passed to the iterator function
--   on its first call.

function pairsFeatures(features)
   return iteratorFeatures, features, 0
end

-- Notice that the pairsFeatures function and the iteratorFeatures function
-- together exactly implement what ipairs(array) does. So these function are
-- not needed. The above code was just to illustrate how to write them
-- if your features are not stored in an array of tensors.

-- Redefine pairsFeatures to use the built-in version.
pairsFeatures = ipairs

-- The targets[i] must yield a number which is the class label. Any object
-- can be passed, provided that either
-- ++ targets is a table, in which case the value at key i is returned or 
--    if it is missing, the value of the index metamethod of targets is
--    returns
-- ++ targets is not a table, in which case the value returned by the 
--    metamethod index is used

-- Here our targets variables is an array, so its a table, and simple indexing
-- will work. So we can pass the targets variable directly.

-- define the LogisticRegression object
logisticRegression = LogisticRegression(features, pairsFeatures, targets)


--------------------------------------------------------------------------------
-- 3. Train the model with stochastic gradient descent

print('')
print('============================================================')
print('Training with SGD')

-- Training the model is by definition finding the optimal weights. How
-- to do this is defined by the table opt.

opt = {
   algo = 'sgd',
   algoParms = {numEpochs=100,
                quiet=false,
                validate=true},
   optimParms = {learningRate = 1e-3,
                 learningRateDecay = 1e-4},
}

print()
print('optimization parameters')
for k,v in pairs(opt) do
   print(k,v)
end


-- train the model
logisticRegression:train(opt)


--------------------------------------------------------------------------------
-- 4. Test the trained model

-- This example provides a simplified test of the model, determining the 
-- probabilities of the various classes for each combination of age and
-- female.

-- return index of largest value
function maxIndex(a,b,c)
   if a >=b and a >= c then return 1 
   elseif b >= a and b >= c then return 2
   else return 3 end
end


-- return predicted brand and the probabilities of each brand
-- for our model
function predictOur(age, female)
   local input = torch.Tensor({age,female})
   local logProbsTensor = logisticRegression:estimate(input) 
   --print('predictOur', age, female, input)
   local probsTensor = torch.exp(logProbsTensor)
   --print('logProbs', logProbs)
   --print('probs', probs[1], probs[2], probs[3] )
   local prob1, prob2, prob3 = probsTensor[1], probsTensor[2], probsTensor[3]
   return maxIndex(prob1, prob2, prob3), prob1, prob2, prob3
end

-- define the table formats
headerFormat = '%3s %6s %5s %5s %5s %s'
detailFormat = '%3d %6d %5.3f %5.3f %5.3f %d'

-- print the header
print(string.format(headerFormat,
                    'age', 'female', 'prob1', 'prob2', 'prob3', 'b'))
-- column b has the best predicted brand

-- print the detail lines
for female = 0,1 do
  for age = 24,38 do
      local predicted, prob1, prob2, prob3 = predictOur(age, female)
      print(string.format(detailFormat,
                          age, female, prob1, prob2, prob3, predicted))
   end
end

--------------------------------------------------------------------------------
-- 5. Continue training with L-BFGS pickup up where SGD left off

print('')
print('============================================================')
print('Restart Training with L-BFGS')
print()

-- define the parameters for L-BFGS
opt = {
   algo = 'lbfgs',
   algoParms= {validate=true},
   optimParms = {lineSearch = optim.lswolfe,
                 verbose = true,
                 maxIter = 100}
}

-- continue training where we left off
logisticRegression:train(opt)

--------------------------------------------------------------------------------
-- 6. Train using only L-BFGS

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
-- 7. Train with L-BFGS in mini batch

print('============================================================')
print('Restart Training with L-BFGS and mini-batches')
print()

print("WRITE ME")