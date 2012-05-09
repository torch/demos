-- logisticregression.lua
-- define class LogisticRegression

-- torch libraries
require 'nn'
require 'optim'

-- local libraries
require 'trainer'
require 'validations'

do
   local LogisticRegression = torch.class('LogisticRegression')

   function LogisticRegression:__init(features, pairsFeatures, targets)
      -- validate parameters
      assert(features, 'features no supplied')
      assert(pairsFeatures, 'pairsFeatures not supplied')
      assert(type(pairsFeatures) == 'function',
             'pairsFeatures not a function; must be an iterator')
      assert(targets, 'targets not supplied')

      -- determine size of model
      local numFeatures, numClasses = 
         Trainer._getCounts(features, pairsFeatures, targets)
      print('LogisticRegression numFeatures', numFeatures)
      print('LogisticRegression numClasses', numClasses)

      -- define model
      self.model = nn.Sequential()
      self.model:add(nn.Linear(numFeatures, numClasses))
      self.model:add(nn.LogSoftMax())

      -- define loss function
      self.criterion = nn.ClassNLLCriterion()

      -- initialize a trainer object
      self.trainer = Trainer(features, pairsFeatures, targets, 
                             self.model, self.criterion)
   end

   function LogisticRegression:estimate(query)
      self.trainer:estimate(query)
   end

   function LogisticRegression:train(opt)
      self.trainer:train(opt)
   end

end -- class LogisticRegression
