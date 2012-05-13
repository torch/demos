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
      validations = Validations()
      assert(features, 'features no supplied')
      assert(targets, 'targets not supplied')
      validations.isIntegerGt0(numClasses, 'numClasses')
      validations.isIntegerGt0(numDimensions, 'numDimensions')

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
      assert(query, 'query not supplied')
      return self.trainer:estimate(query)
   end

   function LogisticRegression:train(nextBatch, opt)
      -- validate presence of parameters
      -- more complicated validation is done in the Trainer:train method
      assert(nextBatch, 'nextBatch not supplied')
      assert(type(nextBatch) == 'function',
             'nextBatch must be a function')

      assert(opt, 'opt not supplied')

   

      self.trainer:train(nextBatch, opt)
   end

end -- class LogisticRegression
