-- trainer.lua
-- define class Trainer

-- torch libraries
require 'nn'
require 'optim'

-- local libraries
require 'Validations'

-- example 1: features and targets are in parallel arrays, batch size is 1
if false then
   -- TODO: Write me
end

-- example: features and arrays are in one table, batch size is 10
if false then 
   -- TODO: WRITE ME
end

do 
   -- Create class object.
   local Trainer = torch.class('Trainer')

   -- Construct the trainer from: 
   -- + inputs    : an object that contains the features from the samples
   --             : See the description of nextBatch, a parameter of the
   --             : train method for how specific inputs and targets are
   --             : extracted from the "inputss" and "targets" parameters.
   -- + targets   : an object that contains the targets from the samples
   -- + model     : a table that behaves just like an nn model
   -- + criterion : a table that behaves just like an nn criterion
   function Trainer:__init(inputs, targets, model, criterion)
      
      -- validations
      Validations.isNotNil(inputs, 'inputs')
      Validations.isNotNil(targets, 'targets')
      Validations.isTable(model, 'model')
      Validations.isTable(criterion, 'criterion')

      -- save the parameters for training
      self.inputs = inputs
      self.targets = targets
      self.model = model
      self.criterion = criterion
   end

   -- Return estimate for the specified query. 
   -- + query: an object of the same type as a feature
   function Trainer:estimate(query)
      Validations.isFunction(self, 'self')
      Validations.isNotNil(query, 'query')

      return self.model:forward(query)
   end

   -- Learn the parameters of the model using one of the optimization method
   -- from the optim package. 
   -- + nextBatch : a "batch iterator" that defines how 
   --             : to iterate over the inputs and targets in batches
   -- + opt       : a table containing the optimization parameters
   -- both of these parameters are described at the end of this file
   function Trainer:train(nextBatch, opt)
      print('Trainer:train opt\n', opt)
      print('Trainer:train self', self)

      -- validate parameters
      Validations.isNotNil(self, 'self')
      Validations.isFunction(nextBatch, 'nextBatch')
      Validations.isNotNil(opt, 'opt')

      if opt.validate == nil or opt.validate then
         Trainer._validateOpt(opt)
      end
      
      -- determine which optimization function to use
      local optimize
      if opt.algo == 'sgd' then
         optimize = optim.sgd
      elseif opt.algo == 'lbfgs' then
         optimize = optim.lbfgs
      else
         error('logic error; opt.algo=' .. opt.algo)
      end

      -- TODO: make these locals after all else works
      x, dl_dx = self.model:getParameters() -- create view of parameters
      
      for epochNumber =1,opt.numEpochs do
         currentLoss = 0  -- TODO: make this local after all else works
         local numBatches = 0
         -- for each batch
         local batchIndices = nextBatch(self.inputs, nil)
         while batchIndices do
            assert(type(batchIndices) == 'table',
                   'nextBatch must return a table or nil')
            --print('train batchIndices', batchIndices)
            numBatches = numBatches + 1

            -- API: Take a single point of evaluation (the parameter vector)
            -- and return loss at that point and the gradient at that point
            -- This implementation returns the average function value and
            -- average gradient over the current batch at the parameter.
            function feval(x_new)
               if x ~= x_new then x:copy(x_new) end
               dl_dx:zero()  -- reset gradient in model
               local cumLoss = 0
               local numInBatch = 0
               -- iterate over the indices in the batch
               for _,nextIndex in pairs(batchIndices) do
                  numInBatch = numInBatch + 1
                  local input = self.inputs[nextIndex]
                  local target = self.targets[nextIndex]
                  Validations.isTensor(input, 'inputs[nextIndex]')
                  Validations.isNumber(target, 'targets[nextIndex]')
                  local lossOnSample = 
                     self.criterion:forward(self.model:forward(input),
                                            target)
                  --print('feval loss', loss, target, input)
                  cumLoss = cumLoss + lossOnSample
                  self.model:backward(input,
                                      self.criterion:backward(
                                         self.model.output,
                                         target))
               end
               return cumLoss / numInBatch, dl_dx / numInBatch
            end -- function feval

            _, fs = optimize(feval, x, opt.optimParams)
            if opt.verboseBatch then
               print('loss values during optimization procedure', fs)
            end
            -- the last value in fs is the value at the optimimum x*
            currentLoss = currentLoss + fs[#fs]
            batchIndices = nextBatch(self.inputs, batchIndices)
         end -- loop over batches

         -- finished with all the batches
         currentLoss = currentLoss / numBatches --?
         if opt.verboseEpoch then
            print(string.format('epoch %d of %d; current loss = %.15f',
                                epochNumber, opt.numEpochs,
                                currentLoss))
         end
         --if epochNumber == 1 then print('epochNumber', epochNumber) halt() end
      end -- for epochNumber
         
   end -- method train

function Trainer._validateOpt(opt)
         
         -- validate opt.algo
         assert(opt.algo, 'opt.algo not supplied')
         assert(opt.algo == 'sgd' or opt.algo == 'lbfgs', 
                'opt.algo must be "sgd" or "lbfgs"')
         
         -- validate opt.numEpochs and supply default
         opt.numEpochs = opt.numEpochs or 100
         Validations.isIntegerGt0(opt.numEpochs,
                                  'opt.numEpochs')

         -- validate opt.verboseBatch and supply default
         if opt.verboseBatch == nil then
            opt.verboseBatch = true
         end
         Validations.isBoolean(opt.verboseBatch,
                               'opt.verboseBatch')

         -- validate opt.verboseEpoch and supply default
         if opt.verboseEpoch == nil then
            opt.verboseEpoch = true
         end
         Validations.isBoolean(opt.verboseEpoch,
                               'op.verboseEpoch')

         -- validate opt.optimParams types and values 
         -- because the function in optim do not do much validation
         -- the optimization function have defaults so don't set defaults here
         if opt.optimParams == nil then
            error('Must supply opt.optimParams even if its nil')
         end
         if opt.algo == 'sgd' then
            Validations.isNilOrNumberGt0(opt.optimParams.learningRate,
                                         'opt.optimParams.learningRate')
            Validations.isNilOrNumberGe0(opt.optimParams.learningRateDecay,
                                         'opt.optimParams.learningRateDecay')
            Validations.isNilOrNumberGe0(opt.optimParams.weightDecay,
                                         'opt.optimParams.weightDecay')
            Validations.isNilOrNumberGe0(opt.optimParams.momentum,
                                         'opt.optimParams.momentum')
            Validations.isNilOrVectorGe0(opt.optimParams.learningRates,
                                         'opt.optimParams.learningRates')
            Validations.isNilOrIntegerGe0(opt.optimParams.evalCounter,
                                          'opt.optimParams.evalCounter')

         elseif opt.algo == 'lbfgs' then
            Validations.isNilOrIntegerGt0(opt.optimParams.maxIter,
                                          'opt.optimParams.maxIter')
            Validations.isNilOrNumberGt0(opt.optimParams.maxEval,
                                         'opt.optimParams.maxEval')
            Validations.isNilOrNumberGe0(opt.optimParams.tolFun,
                                         'opt.optimParams.tolFun')
            Validations.isNilOrNumberGe0(opt.optimParams.tolX,
                                         'opt.optimParams.tolX')
            Validations.isNilOrFunction(opt.optimParams.lineSearch,
                                        'opt.optimParams.lineSearch')
            Validations.isNilOrIntegerGe0(opt.optimParams.learningRate,
                                          'opt.optimParams.learningRate')
            Validations.isNilOrBoolean(opt.optimParams.verbose,
                                       'opt.optimParams.verbose')

         else
            error('logic error; opt.algo=' .. opt.algo)
         end
      end -- of validations

--[[ Description of batch iterators.

The constructor for Trainer requires a parameter called nextBatch. It's 
a "batch iterator." This section explains what a batch iterator is and
how to write one.

The purposes of nextBatch are to 

- Define to the optimization procedure how to pull individual inputs
  and targets out of the inputs and targets parameters. This allows
  you to use many different data structures. For example, you can use
  parallel arrays for inputs and targets. In that case, inputs[i]
  corresponds to targets[i]. As another example, you can use a 2D
  tensor to hold both the inputs and targets. In that case, you
  might have a rows correspond to a sample where the first column is
  the target. Or you might decided that a column corresponds to a
  sample and the target is in the last row.

- Define batches, which are sequences of training samples (inputs and 
  targets that are used in one stage of optimization. Some optimization
  procedures do not use batch and some require batches.

The batch iterator nextBatch defines both the data structures and
batches.

The API for nextBatch is similar to that of Lua's next function, except that
the keys are tables of indices and only one value is returned. The
nextBatch(inputs, keys) function has this API:

+ inputs : any Torch object (often a table or 2D tensor). This is 
             the same inputs object passed to the Trainer's constructor.
+ keys   : either nil or a table of indices for inputs.
+ result : if keys == nil then
             result is a table containing the initial indices of the
             inputs. The inputs and targets are retrieved by
             the Trainer through this iteration:
           if keys represents the last portion of the batch then
             result is nil
           else
             result is a table containing the next set of keys
           end
   
The iteration employed in the training loop works like this:

   local batchIndices = nextBatch(inputs, nil)
   while batchIndices do
      for _,index in ipairs(batchIndices) do
          local input = inputs[index]
          local target = targets[index]
          <learn using feature (a 1D tensor)  and target (a number)>
      end -- iteration over elements of the batch
      batchIndices = nextBatch(inputs, batchIndices)                 
   end -- iteration over batches               

--]]

--[[ opt table structure

opt is a table with these fields and [default] values

+ opt.algo         : string, name of the optim algorithm. One of
                     "sgd" (stochastic gradient descent)
                     "lbfgs" (L-BFGS)
+ opt.numEpochs    : integer [100] > 0, number of epochs
+ opt.optimParams  : table, parameters for the optim function
                     described below
+ opt.validate     : boolean [true], if true, the fields in
                     opt.optimParams are checked for type and 
                     reasonable values
+ opt.verboseBatch : boolean[true], if true, loss of
                     at each point in the batch is printed
+ opt.verboseEpoch : boolean [true], if true, loss for
                     each epoch is printed

The table opt.optimParams is passed directly to the optimization
function, but first its fields are validated if opt.algoParms.validate
== true.

Most of the optimParams have default values that are documented in the
optim function. We recommend that you do not rely on default values,
as the defaults may change when algorithms or implementations are
enhanced.
   
The validations if opt.algo == 'sgd' are these:

+ opt.optimParams.learningRate      : nil or number > 0
+ opt.optimParams.learningRateDecay : nil or number >= 0 
+ opt.optimParams.weightDecay       : nil or number >= 0
+ opt.optimParams.momentum          : nil or number >= 0
+ opt.optimParams.learningRates     : nil or vector of number >= 0
+ opt.optimParams.evalCounter       : nil or integer >= 0
  
The validations if opt.algo == 'lbfgs' are these:

+ opt.optimParams.maxIter      : nil or integer > 0
+ opt.optimParams.maxEval      : nil or number > 0
+ opt.optimParams.tolFun       : nil or number >= 0
+ opt.optimParams.tolX         : nil or number >= 0
+ opt.optimParams.lineSearch   : nil or a function
+ opt.optimParams.learningRate : nil or integer >= 0
+ opt.optimParams.verbose      : nil or boolean

--]]

end -- class Trainer
                                     