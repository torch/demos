-- LinearRegression-test.lua
-- unit test of class LinearRegression

require 'LinearRegression'

myTests = {}

tester = torch.Tester()

-- return index of highest unnormalized probability
local function selectMostLikely(a, b, c)
   if     a >= b and a >= c then return 1
   elseif b >= a and b >= c then return 2
   else                          return 3
   end
end


-- generate data
-- y = 20 + 40*feature1 - 30*feature2
-- feature ~ Uniforma[0,1)
-- 1000 observations
function generateData()

   -- define true model
   local function estimate(feature1, feature2)
      return 20 + 40 * feature1 - 30 * feature2
   end

   local inputs = {}
   local targets = {}
   local numClasses = 3
   local numDimensions = 2
   local numSamples = 10
   for i=1,numSamples do
      local feature1 = math.random() -- uniform pseudo-random in [0,1)
      local feature2 = math.random()
      local targetValue = estimate(feature1, feature2)

      input = torch.Tensor(2)
      input[1] = feature1
      input[2] = feature2
      inputs[#inputs + 1] = input
      target = torch.Tensor(1)
      target[1] = targetValue
      targets[#targets + 1] = target
   end
   return inputs, targets, numDimensions
end

-- test accuracy vs. known results, return fraction correct
-- count as correct if within 10%
function testEstimates(name, lr, inputs, targets)
   local countCorrect = 0
   for i,input in ipairs(inputs) do
      local estimate = (lr:estimate(input))[1]
      local expected = targets[i][1]
      local ratio = estimate / expected
      print('testEstimate', estimate, expected, ratio)
      if 0.90 <= ratio and ratio <= 1.10 then 
         countCorrect = countCorrect + 1 
      end
   end
   local fractionCorrect = countCorrect / #inputs
   print('fraction correct', name, fractionCorrect)
   return fractionCorrect
end

function runTest(nextBatch, opt, requiredAccuracy)
   local inputs, targets, numDimensions = generateData()

   local lr = LinearRegression(inputs, targets, numDimensions)

   print(' Starting training of', opt.algo)
   lr:train(nextBatch, opt)

   local accuracy = testEstimates(opt.algo, lr, inputs, targets)
   tester:assertge(accuracy, 
                   requiredAccuracy, 
                   opt.algo .. ' result not expected')
end

-- batch size is all samples
function nextBatchAll(inputs, prevIndices)  -- batch size is all
   if prevIndices == nil then
      local result = {}
      for i,_ in ipairs(inputs) do
         result[#result + 1] = i
      end
      return result
   else
      return nil
   end
end
   
-- batch size is one sample
function nextBatchOne(inputs, prevIndices)  -- batch size is 1
   if prevIndices == nil then
      return {1}
   else
      local onlyIndex = prevIndices[1]
      if onlyIndex >= #inputs then
         return nil
      else
         return {onlyIndex + 1}
      end
   end
end
   
--------------------------------------------------------------------------------
-- define test cases
--------------------------------------------------------------------------------

function myTests.test()
   if true then return end

   opt = {algo = 'cg',
          numEpochs = 3,  -- does not converge with 2 epochs
          verboseBatch = false,
          optimParams = {} -- use defaults
   }

   -- CLEMENT: Does CG accept a batch size of 1?
   runTest(nextBatchOne, opt, 0.9)
end

function myTests.testLbfgs()
   if true then return end

   opt = {algo = 'lbfgs',
          numEpochs = 1,
          optimParams = {} -- use defaults
   }

   runTest(nextBatchAll, opt, 0.9)
end

function myTests.testSgd()
   --if true then return end
   opt = {algo = 'sgd',
          verboseBatch = false,
          numEpochs = 12,  -- does not converge with 11 epochs
          optimParams = {learningRate = 1e-1}
   }

   runTest(nextBatchOne, opt, 0.9)
end

--------------------------------------------------------------------------------
-- run test cases
-------------------------------------------------------------------------------

tester:add(myTests)
tester:run()
