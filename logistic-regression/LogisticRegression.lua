-- logisticregression.lua
-- define class LogisticRegression

-- torch libraries
require 'nn'
require 'optim'

-- local libraries
require 'Trainer'
require 'Validations'

do
   local LogisticRegression = torch.class('LogisticRegression')

   function LogisticRegression:__init(features, targets)
      -- validate parameters
      assert(features, 'features no supplied')
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
      assert(query, 'query not supplied')
      return self.trainer:estimate(query)
   end

   function LogisticRegression:train(opt, pairsFeatures)
      -- validate presence of parameters
      -- more complicated training is done in the Trainer:train method
      assert(opt, 'opt not supplied')

      assert(pairsFeatures, 'pairsFeatures not supplied')
      print(type(pairsFeatures))
      assert(type(pairsFeatures) == 'function',
             'pairsFeature must be a function returning an iterator')
   

      self.trainer:train(opt, pairsFeatures)
   end

end -- class LogisticRegression
