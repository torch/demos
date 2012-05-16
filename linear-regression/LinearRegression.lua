-- LinearRegression.lua
-- define class LinearRegression

-- torch libraries
require 'nn'
require 'optim'

-- local libraries
require 'Trainer'
require 'Validations'

do
   local LinearRegression = torch.class('LinearRegression')

   function LinearRegression:__init(inputs, targets,  
                                    numDimensions)
      -- validate parameters
      Validations.isTable(self, 'self')
      Validations.isNotNil(inputs, 'inputs')
      Validations.isNotNil(targets, 'targets')
      Validations.isIntegerGt0(numDimensions, 'numDimensions')

      -- save data for type validations
      self.inputs = inputs
      self.targets = targets


      -- define model
      self.model = nn.Sequential()
      local numOutputs = 1
      self.model:add(nn.Linear(numDimensions, numOutputs))

      -- define loss function
      self.criterion = nn.MSECriterion()

      -- initialize a trainer object
      self.trainer = Trainer(inputs, targets,
                             self.model, self.criterion)
   end

   function LinearRegression:estimate(query)
      -- validate parameters
      Validations.isTable(self, 'self')
      Validations.isNotNil(query, 'query')

      return self.trainer:estimate(query)
   end

   function LinearRegression:getCriterion()
      Validations.isTable(self, 'self')
      return self.trainer:getCriterion()
   end

   function LinearRegression:getModel()
      Validations.isTable(self, 'self')
      return self.trainer:getModel()
   end

   function LinearRegression:train(nextBatch, opt)

      -- validate parameters
      Validations.isTable(self, 'self')
      Validations.isFunction(nextBatch, 'nextBatch')
      assert(opt, 'opt not supplied') -- more validation is done in Trainer

      -- validate that an input is a Tensor and a target is a Tensor
      local indices = nextBatch(self.inputs, nil)
      for k, index in pairs(indices) do
         Validations.isTensor(self.inputs[index], '1st input')
         Validations.isTensor(self.targets[index], '1st target')
         break
      end

      self.trainer:train(nextBatch, opt)
   end

end -- class LinearRegression
