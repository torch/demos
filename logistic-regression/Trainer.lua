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
   -- + features  : an object that contains the features from the samples
   --             : See the description of nextBatch, a parameter of the
   --             : train method for how specific features and targets are
   --             : extracted from the "features" and "targets" parameters.
   -- + targets   : an object that contains the targets from the samples
   -- + model     : a table that behaves just like an nn model
   -- + criterion : a table that behaves just like an nn criterion
   function Trainer:__init(features, targets, model, criterion)
      
      -- check that parameters are supplied and have plausible types
      assert(features, 'features no supplied')

      assert(targets, 'targets not supplied')

      assert(model, 'model not supplied')
      assert(type(model) == 'table', 'model is not a table')

      assert(criterion, 'criterion not supplied')
      assert(type(criterion) == 'table', 'criterion is not a table')

      -- save the parameters for training
      self.features = features
      self.pairsFeatures = pairsFeatures
      self.targets = targets
      self.model = model
      self.criterion = criterion

      print('__init self', self)
      
   end

   -- Return estimate for the specified query. 
   -- + query: an object of the same type as a feature
   function Trainer:estimate(query)
      return self.model:forward(query)
   end

   -- Learn the parameters of the model using one of the optimization method
   -- from the optim package. 
   -- + nextBatch : a "batch iterator" which is a function of two parameter
   --             : returning one or two results with the API given below.
   -- + opt       : a table containing the optimization parameters

--[[ batch iterator/nextBatch API

The API for nextBatch is similar to that of Lua's next function, except that
the keys are tables of indices and only one value is returned. The
nextBatch(features, keys) function has this API:

+ features : any Torch object (often a table or 2D tensor). This is 
             the same features object passed to the Trainer's constructor.
+ keys     : either nil or a table of indices for features.
+ result   : if keys == nil then
               result is a table containing the initial indices of the
               features. The features and targets are retrieved by
               the Trainer through this iteration:
             if keys represents the last portion of the batch then
               result is nil
             else
               result is a table containing the next set of keys
             end
   
The iteration employed in the training loop works like this:

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

   -- Train the model using the criterion specified in construction,
   -- the samples accessed through the nextBatch function, and the
   -- options in table opt.
   function Trainer:train(nextBatch, opt)
      print('Trainer:train nextBatch', nextBatch)
      print('Trainer:train opt\n', opt)
      print('Trainer:train self', self)
      
      validations = Validations()
      
      -- validate nextBatch
      assert(nextBatch, 'nextBatch not supplied')
      assert(type(nextBatch) == 'function',
             'nextBatch not a function')
      
      -- validate opt
      assert(opt,'opt not supplied')
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
         local batchIndices = nextBatch(self.features, nil)
         while batchIndices do
            assert(type(batchIndices) == 'table',
                   'nextBatch must return a table or nil')
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
                  local lossOnSample = 
                     self.criterion:forward(
                         self.model:forward(self.features[nextIndex]),
                         self.targets[nextIndex])
                     --print('feval loss', loss, self.targets[nextIndex],
                     --      self.features[nextIndex])
                  cumLoss = cumLoss + lossOnSample
                  self.model:backward(self.features[nextIndex],
                                      self.criterion:backward(
                                         self.model.output,
                                         self.targets[nextIndex]))
               end
               return cumLoss / numInBatch, dl_dx / numInBatch
            end -- function feval

            _, fs = optimize(feval, x, opt.optimParams)
            if opt.verboseBatch then
               print('loss values during optimization procedure', fs)
            end
            -- the last value in fs is the value at the optimimum x*
            currentLoss = currentLoss + fs[#fs]
            batchIndices = nextBatch(self.features, batchIndices)
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
         validations.isIntegerGt0(opt.numEpochs,
                                  'opt.numEpochs')

         -- validate opt.verboseBatch and supply default
         if opt.verboseBatch == nil then
            opt.verboseBatch = true
         end
         validations.isBoolean(opt.verboseBatch,
                               'opt.verboseBatch')

         -- validate opt.verboseEpoch and supply default
         if opt.verboseEpoch == nil then
            opt.verboseEpoch = true
         end
         validations.isBoolean(opt.verboseEpoch,
                               'op.verboseEpoch')

         -- validate opt.optimParams types and values 
         -- because the function in optim do not do much validation
         -- the optimization function have defaults so don't set defaults here
         if opt.optimParams == nil then
            error('Must supply opt.optimParams even if its nil')
         end
         if opt.algo == 'sgd' then
            validations.isNilOrNumberGt0(opt.optimParams.learningRate,
                                         'opt.optimParams.learningRate')
            validations.isNilOrNumberGe0(opt.optimParams.learningRateDecay,
                                         'opt.optimParams.learningRateDecay')
            validations.isNilOrNumberGe0(opt.optimParams.weightDecay,
                                         'opt.optimParams.weightDecay')
            validations.isNilOrNumberGe0(opt.optimParams.momentum,
                                         'opt.optimParams.momentum')
            validations.isNilOrVectorGe0(opt.optimParams.learningRates,
                                         'opt.optimParams.learningRates')
            validations.isNilOrIntegerGe0(opt.optimParams.evalCounter,
                                          'opt.optimParams.evalCounter')

         elseif opt.algo == 'lbfgs' then
            validations.isNilOrIntegerGt0(opt.optimParams.maxIter,
                                          'opt.optimParams.maxIter')
            validations.isNilOrNumberGt0(opt.optimParams.maxEval,
                                         'opt.optimParams.maxEval')
            validations.isNilOrNumberGe0(opt.optimParams.tolFun,
                                         'opt.optimParams.tolFun')
            validations.isNilOrNumberGe0(opt.optimParams.tolX,
                                         'opt.optimParams.tolX')
            validations.isNilOrFunction(opt.optimParams.lineSearch,
                                        'opt.optimParams.lineSearch')
            validations.isNilOrIntegerGe0(opt.optimParams.learningRate,
                                          'opt.optimParams.learningRate')
            validations.isNilOrBoolean(opt.optimParams.verbose,
                                       'opt.optimParams.verbose')

         else
            error('logic error; opt.algo=' .. opt.algo)
         end
      end -- of validations

end -- class Trainer
                                     