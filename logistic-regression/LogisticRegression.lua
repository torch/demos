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

   function LogisticRegression:__init(features, targets,  
                                      numClasses, numDimensions)
      -- validate parameters
      Validations.isTable(self, 'self')
      Validations.isNotNil(features, 'features')
      Validations.isNotNil(targets, 'targets')
      Validations.isIntegerGt0(numClasses, 'numClasses')
      Validations.isIntegerGt0(numDimensions, 'numDimensions')

      -- define model
      self.model = nn.Sequential()
      self.model:add(nn.Linear(numDimensions, numClasses))
      self.model:add(nn.LogSoftMax())

      -- define loss function
      self.criterion = nn.ClassNLLCriterion()

      -- initialize a trainer object
      self.trainer = Trainer(features, targets,
                             self.model, self.criterion)
   end

   function LogisticRegression:estimate(query)
      -- validate parameters
      Validations.isTable(self, 'self')
      Validations.isNotNil(query, 'query')

      return self.trainer:estimate(query)
   end

   function LogisticRegression:getCriterion()
      Validations.isTable(self, 'self')
      return self.trainer:getCriterion()
   end

   function LogisticRegression:getModel()
      Validations.isTable(self, 'self')
      return self.trainer:getModel()
   end

   function LogisticRegression:train(nextBatch, opt)

      -- validate parameters
      Validations.isTable(self, 'self')
      Validations.isFunction(nextBatch, 'nextBatch')
      assert(opt, 'opt not supplied') -- more validation is done in Trainer

      self.trainer:train(nextBatch, opt)
   end

end -- class LogisticRegression
