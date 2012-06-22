-- LogisticRegression-test.lua
-- unit test of class LogisticRegression

require 'LogisticRegression'

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
-- prob(1) propTo exp(10*feature1 + 20*feature2)
-- prob(2) propTo exp(20*feature1 + 10*feature2)
-- prob(3) propTo exp( 1*feature1 +  1*feature2)
-- feature ~ Uniform[0,1)
-- 1000 observations
function generateData()

   -- define unnormalized probabilities
   local function uprob1(feature1, feature2)
      return math.exp(10*feature1 + 20*feature2)
   end
   local function uprob2(feature1, feature2)
      return math.exp(20*feature1 + 10*feature2)
   end
   local function uprob3(feature1, feature2)
      return math.exp(return1 + feature2)
   end

   local inputs = {}
   local targets = {}
   local numClasses = 3
   local numDimensions = 2
   local numSamples = 10
   for i=1,numSamples do
      local feature1 = math.random() -- uniform pseudo-random in [0,1)
      local feature2 = math.random()
      local target = selectMostLikely(uprob1(feature1, feature2),
                                      uprob2(feature1, feature2),
                                      uprob2(feature1, feature2))
      input = torch.Tensor(2)
      input[1] = feature1
      input[2] = feature2
      inputs[#inputs + 1] = input
      targets[#targets + 1] = target
   end
   return inputs, targets, numClasses, numDimensions
end

-- test accuracy vs. known results, return fraction correct
function testEstimates(name, lr, inputs, targets)
   local countSame = 0
   for i,input in ipairs(inputs) do
      local uprobs = lr:estimate(input) -- returns vector on unnormalized probs
      local estimate = selectMostLikely(uprobs[1], uprobs[2], uprobs[3])
      local expected = targets[i]
      if estimate == expected then 
         countSame = countSame + 1 
      end
   end
   local fractionSame = countSame / #inputs
   print('fraction same', name, fractionSame)
   return fractionSame
end

function runTest(nextBatch, opt, requiredAccuracy)
   local inputs, targets, numClasses, numDimensions = generateData()

   local lr = LogisticRegression(inputs, targets, numClasses, numDimensions)

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

function myTests.testCg()

   opt = {algo = 'cg',
          numEpochs = 100,  -- does not always work with 10 epochs
          verboseBatch = false,
          optimParams = {} -- use defaults
   }

   -- CLEMENT: Does CG accept a batch size of 1?
   runTest(nextBatchOne, opt, 0.9)
end

function myTests.testLbfgs()

   opt = {algo = 'lbfgs',
          numEpochs = 5,
          optimParams = {} -- use defaults
   }

   runTest(nextBatchAll, opt, 0.9)
end

function myTests.testSgd()
   opt = {algo = 'sgd',
          numEpochs = 1000,
          optimParams = {}
   }

   runTest(nextBatchOne, opt, 0.9)
end

--------------------------------------------------------------------------------
-- run test cases
-------------------------------------------------------------------------------

tester:add(myTests)
tester:run()
