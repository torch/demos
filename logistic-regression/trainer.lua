-- trainer.lua
-- define class Trainer

-- torch libraries
require 'nn'
require 'optim'

-- local libraries
require 'validations'

-- example: read UCLA data set
-- TODO: rewrite to use to opt table
if false then
   -- setup to handle the UCLA data set
   local features = torch.Tensor(735,3)  -- 735 observations of 3 dimensions
   local targets = torch.Tensor(735)     -- also 735 observations
   initialize(features, targets)         -- you supply this function
   
   -- this function is an iterator for the elements in the first dimension
   -- of a tensor
   -- see PiL version 2, p 59 for details
   -- + tensor: a 2D tensor in this example, though the function works
   --   for any dimension
   -- + index: the previous value returned was tensor[index]
   local function iterTensor(tensor, index)
      index = index + 1
      if index > tensor:size(1) then
         return nil
      else
         return index, tensor[index]
      end
   end

   -- like pairs(array) but for a tensor
   -- + features: a 2D tensor such that each row is an observation
   local function myPairsFeatures(features)
      return iterTensor, tensor, 0
   end

   -- optimize using SGD
   local optim = {
      method = 'sgd',
      quiet=false,
      params = {  -- parameters for optim.sgd
         learningRate = 1e-3,
         learningRateDecay = 1e-4,
      }
   }

   local logreg = LogisticRegression()
   logreg:train(features, myPairsFeatures, targets, optim)
   
   local query = makeQuery()  -- you supply this
   local classEstimate = logreg:estimate(query)
   -- do something with the estimate
end

-- example: instead of a single sample, iterate over mini batches
do
   -- WRITE ME
end

do -- TODO: fix example to match new init code
   -- create class object
   local Trainer = torch.class('Trainer')

   -- initializer: define the data
   -- + features      : an object that is passed to pairsFeatures
   -- + pairsFeatures : a function taking features as an argument an returning
   --                   an iterator; similar to the built-in pairs(table)
   --                   except that in addition to return an key and value,
   --                   it can return an array of keys and a corresponding
   --                   array of values. 
   -- + targets       : an object that when indexed by a key returns a value,
   --                   the target value for the observations with the key

   -- Example 1: features and targets are in lua arrays
   -- features[i] for integer i is a tensor
   -- targets[i] is a number, the class number
   -- pairFeatures() is a function with two arguments with two results
   -- + i        : a number, the index of the last returned sample
   -- + features : the features array
   -- + result_1 : next index, so that features == features[result_1] and
   --              targets[result_1] is a number, the next class number
   -- + result_2 : next feature, a tensor

   -- Note that in this example the class identifiers are integers, but they
   -- can be any value at all that implements the == test in a way that is
   -- appropriate for your data

   -- Example 2: features are tensors
   -- features is a 2D tensor where features[i] is the i-th sample, a tensor
   -- targets is a 1D tensor where targets[i] is the i-th label, a number
   -- pairsFeatures() is a function of two arguments with two results:
   -- + i        : a number, the index of the last returned sample
   -- + features : the 2D tensor with all the samples
   -- + result_1 : next index, so that features == features[result_1] and
   --              targets[result_1] is a number, the next class number
   -- + result_2 : next feature, a tensor

   -- Example 3: features are tensors and the training proceeds in mini batches
   -- features is a 2D tensor where features[i] is the i-th sample, a tensor
   -- targets is a 1D tensor where targets[i] is the i-th label, a number
   -- pairsFeatures() is a function of two arguments with two results:
   -- + iArray   : an array of  numbers, 
   --              each element is an index for a sample
   -- + features : the 2D tensor with all the samples
   -- + result_1 : an array of indices; the targets are
   --              targets[result_1[1]], targets[result_1[2]], ...
   -- + result_2 : an array of features, each a tensor
   --              the optimization evalution function
   --              averages the values from forward and backward 
   --              propagating each of the features
   function Trainer:__init(features, pairsFeatures, targets, model, criterion)
      
      -- check that parameters are supplied and have plausible types
      assert(features, 'features no supplied')

      assert(pairsFeatures, 'pairsFeatures not supplied')
      assert(type(pairsFeatures) == 'function',
             'pairsFeatures not a function; must return an iterator')

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

   -- return tensor of log probabilities for each class label 
   -- + query: an object of the same type as a feature
   function Trainer:estimate(query)
      return self.model:forward(query)
   end

   -- learn the parameters of the model using one of the optimization method
   -- from the optim package. Takes one parameter opt, a table, with these
   -- fields.
   -- + opt.algo       : string, name of the optim algorithm. One of
   --                    "sgd" (stochastic gradient descent)
   --                    "lbfgs" (L-BFGS)
   -- + opt.algoParms  : table, parameters for the particular algorithm
   -- + opt.optimParms : table, parameters for the optim function

   -- if opt.algo == "sgd" these fields and [defaults] in opt.algoParms are used
   -- + opt.algoParms.numEpochs : integer [100] > 0, number of epochs
   -- + opt.algoParms.quiet     : boolean [false], if true, the average loss for
   --                             each epoch is printed
   -- + opt.algoParms.validate  : boolean [true], if true, the fields in
   --                             opt.optimParms are checked for type and 
   --                             reasonable values

   -- if opt.algo == "lbfgs" these fields in opt.algoParms are used.
   -- + opt.algoParms.validate  : boolean [true], if true, the fields in
   --                             opt.optimParms are checked for type and 
   --                             reasonable values


   -- The table opt.optimParms is passed directly to the optimization
   -- function, but first its fields are validated if opt.algoParms.validate =
   -- true.

   -- Most of the optimParms have default values that are documented 
   -- in the optim function. We recommend that 
   -- you do not rely on default values, as the defaults may change when
   -- algorithms or implementations are enhanced.
   
   -- The validations if opt.algo == 'sgd' are these:
   -- + opt.optimParms.learningRate      : nil or number > 0
   -- + opt.optimParms.learningRateDecay : nil or number >= 0 
   -- + opt.optimParms.weightDecay       : nil or number >= 0
   -- + opt.optimParms.momentum          : nil or number >= 0
   -- + opt.optimParms.learningRates     : nil or vector of number >= 0
   -- + opt.optimParms.evalCounter       : nil or integer >= 0
  
   -- The validations if opt.algo == 'lbfgs' are these:
   -- + opt.optimParms.maxIter      : nil or integer > 0
   -- + opt.optimParms.maxEval      : nil or number > 0
   -- + opt.optimParms.to1Fun       : nil or number >= 0
   -- + opt.optimParms.to1X         : nil or number >= 0
   -- + opt.optimParms.lineSearch   : nil or a function
   -- + opt.optimParms.learningRate : nil or integer >= 0
   -- + opt.optimParms.verbose      : nil or boolean
   function Trainer:train(opt)
      print('train opt\n', opt)
      print('train self', self)

      validations = Validations()

      -- validate opt.algo and set algo
      assert(opt,'opt not supplied')
      assert(opt.algo, 'opt.algo not supplied')
      assert(opt.algo == 'sgd' or
             opt.algo == 'lbfgs', 
             'opt.algo must be "sgd" or "lbfgs"')

      -- validate opt.algoParms
      validations.isNotNil(opt.algoParms, 'opt.algoParms')

      -- TODO: verify that each of these parameters is used
      -- TODO: eliminate one set of parameters if merge works
      if opt.algo == 'sgd' then
         opt.algoParms = opt.algoParms or {numEpochs = 100,
                                           quiet = false,
                                           validate = true}
         validations.isIntegerGt0(opt.algoParms.numEpochs,
                                  'opt.algoParms.numEpochs')
         validations.isBoolean(opt.algoParms.quiet,
                               'opt.algoParms.quiet')
         validations.isBoolean(opt.algoParms.validate,
                               'op.algoParms.validate')
         validations.isIntegerGt0(opt.algoParms.numEpochs,
                                  'opt.algoParms.numEpochs')
         validations.isBoolean(opt.algoParms.quiet,
                               'opt.algoParms.quiet')
         validations.isBoolean(opt.algoParms.validate,
                               'opt.algoParms.validate')

      elseif opt.algo == 'lbfgs' then
         opt.algoParms = opt.algoParms or {validate = true}
         opt.algoParms.numEpochs = opt.algoParms.numEpochs or 1
         validations.isBoolean(opt.algoParms.validate,
                               'opt.algoParms.validate')

      else
         error('logic error; opt.algo=' .. opt.algo)
      end

      -- validate opt.optimParms
      if opt.algo == 'sgd' then
         validations.isNilOrNumberGt0(opt.optimParms.learningRate,
                          'opt.optimParms.learningRate')
         validations.isNilOrNumberGe0(opt.optimParms.learningRateDecay,
                                      'opt.optimParms.learningRateDecay')
         validations.isNilOrNumberGe0(opt.optimParms.weightDecay,
                                      'opt.optimParms.weightDecay')
         validations.isNilOrNumberGe0(opt.optimParms.momentum,
                                      'opt.optimParms.momentum')
         validations.isNilOrVectorGe0(opt.optimParms.learningRates,
                                      'opt.optimParms.learningRates')
         validations.isNilOrIntegerGe0(opt.optimParms.evalCounter,
                                       'opt.optimParms.evalCounter')

      elseif opt.algo == 'lbfgs' then
         validations.isNilOrIntegerGt0(opt.optimParms.maxIter,
                                       'opt.optimParms.maxIter')
         validations.isNilOrNumberGt0(opt.optimParms.maxEval,
                                      'opt.optimParms.maxEval')
         validations.isNilOrNumberGe0(opt.optimParms.to1Fun,
                                      'opt.optimParms.to1Fun')
         validations.isNilOrNumberGe0(opt.optimParms.to1X,
                                      'opt.optimParms.to1X')
         validations.isNilOrFunction(opt.optimParms.lineSearch,
                                     'opt.optimParms.lineSearch')
         validations.isNilOrIntegerGe0(opt.optimParms.learningRate,
                                       'opt.optimParms.learningRate')
         validations.isNilOrBoolean(opt.optimParms.verbose,
                                    'opt.optimParms.verbose')

      else
         error('logic error; opt.algo=' .. opt.algo)
      end

      -- attempt to converge the two methods into one block of code
      
      local optimization
      if opt.algo == 'sgd' then
         optimization = optim.sgd
      elseif opt.algo == 'lbfgs' then
         optimization = optim.lbfgs
      else
         error('logic error; opt.algo=' .. opt.algo)
      end
      
      x, dl_dx = self.model:getParameters() -- create view of parameters
      
      for epochNumber = 1,opt.algoParms.numEpochs do
         local numSamples = 0
         local cumLoss = 0
         for sampleIndices, samples in pairsFeatures(self.features) do
            -- determine average loss and gradient over the mini batch
            -- which may have size one

            -- if samplesIndices and samples are tables, then they define
            -- the mini batch. If they are not tables, then turn them into
            -- single entry arrays so that one code base will work on 
            -- both cases.
            if (type(sampleIndices) ~= 'table') and
               (type(samples) ~= 'table') then
               -- turn each into an array
               sampleIndices = {sampleIndices}
               samples = {samples}
            end

            -- build array of targets
            targets = {}
            for _,sampleIndex in ipairs(sampleIndices) do
               targets[#targets + 1] = self.targets[sampleIndex]
            end

            if false then
               -- print the mini batch
               print('mini batch')
               for i = 1,#samples do
                  print(i, samples[i], targets[i])
               end
            end

            -- return as values the averages of the mini batch
            local function feval(x_new)
               if x ~= x_new then
                  x:copy(x_new)
               end
               dl_dx:zero()
               local cumLoss = 0
               for i=1,#samples do
                  local loss_x =
                     self.criterion:forward(self.model:forward(samples[i]),
                                            targets[i])
                  cumLoss = cumLoss + loss_x
                  self.model:backward(samples[i],
                                      self.criterion:backward(self.model.output,
                                                              targets[i]))
               end

               return cumLoss / #samples, dl_dx / #samples
            end -- function feval
 
            _, fs = optimization(feval, x, opt.optimParms)
            if not opt.algoParms.quiet then
               print('losses on mini batch:', fs)
               --print(fs)
            end
         end -- for sampleIndices, samples
      end -- for epochNumber

      if false then -- dead code
         if opt.algo == 'sgd' then
            return self:_trainSgd(opt)
         elseif opt.algo == 'lbfgs' then
            return self:_trainLbfgs(opt)
         else
            error('logic error: opt.algo=' .. opt.algo)
         end
      end
   end -- method train

   -- run L-BFGS algorithm; mutate self.model to contain updated parameters
   -- private method
   function Trainer:_trainLbfgs(opt)
      print('trainLbfgs opt', opt)
      print('trainLbfgs self', self)

      x, dl_dx = self.model:getParameters()  -- create view of parameters
      
      function feval(x_new)
         if x ~= x_new then
            x:copy(x_new)
         end
         
         dl_dx:zero() -- reset gradients
         
         local numSamples = 0
         local loss_x = 0
         for sampleIndex, sample in pairsFeatures(self.features) do
            numSamples = numSamples + 1
            local target = self.targets[sampleIndex]
            loss_x = 
               loss_x + self.criterion:forward(self.model:forward(sample), 
                                               target)
            self.model:backward(sample,
                                self.criterion:backward(self.model.output,
                                                        target))
         end
         
         -- average over the batch
         loss_x = loss_x / numSamples
         dl_dx = dl_dx:div(numSamples)

         return loss_x, dl_dx
      end -- function eval

      _, fs = optim.lbfgs(feval, x, opt.optimParms)

      if not optAlgo.quiet then
         print('history of L-BFGS evaluations:')
         print(fs)
      end
   end -- function trainLbfgs

   -- run SGD algorithm; mutate self.model to contain updated parameters
   -- private method
   function Trainer:_trainSgd(opt)
      print('trainSgd opt', opt) 
      print('trainSgd self', self)

      x, dl_dx = self.model:getParameters()  -- create view of parameters
  
      -- an epoch is a full cycle over the training samples
      for epochNumber = 1,opt.algoParms.numEpochs do
      
         -- determine average loss over entire training set
         local numSamples = 0
         local cumLoss = 0
         for sampleIndex, sample in pairsFeatures(self.features) do
            numSamples = numSamples + 1
            target = self.targets[sampleIndex]
            --print('trainSgd sampleIndex', sampleIndex)
            --print('trainSgd sample', sample)
            --print('trainSgd target', target)
            
            -- nest feval so that it can easily access sample and target
            function feval(x_new)
               if x ~= x_new then
                  x:copy(x_new)
               end
               dl_dx:zero()  -- reset gradient in the model
               local loss_x =  
                  self.criterion:forward(self.model:forward(sample), 
                                         target)
               self.model:backward(sample,
                                   self.criterion:backward(self.model.output, 
                                                           target))
               return loss_x, dl_dx
            end -- function feval
            
            _, fs = optim.sgd(feval, x, opt.optimParms)
            cumLoss = cumLoss + fs[1]
         end -- for sampleIndex, sample

         if not opt.quiet then
            print(string.format('epoch %d of %d: average loss = %f',
                                epochNumber, 
                                opt.algoParms.numEpochs, 
                                cumLoss / numSamples))
         end
      end -- for epochNumber
   end -- function trainSgd


   -- return number of features and number of classes
   -- private
   function Trainer._getCounts(features, pairsFeatures, targets)
      -- determine number of features and all possible target values
      local numFeatures = 0
      local targetSet = {}
      local firstTime = true
      for i, feature in pairsFeatures(features) do
         if firstTime then
            assert(type(feature) == 'userdata',
                  'features element is not a Tensor')
            assert(feature:nDimension() == 1,
                   'features element is not a 1D Tensor')
            numFeatures = feature:size(1)
            assert(numFeatures, 'features element is not a Tensor')
            firstTime = false
         end
         targetSet[targets[i]] = true
      end

      -- count the number of elements in targetSet
      -- the elements are not necessarily adjacent to each other
      -- the elements are not necessarily integers or even numeric
      local numClasses = 0
      for _,_ in pairs(targetSet) do
         numClasses = numClasses + 1
      end
      print('_getCounts results', numFeatures, numClasses)
      return numFeatures, numClasses
   end -- function Trainer._getCounts
end -- class Trainer
                                     