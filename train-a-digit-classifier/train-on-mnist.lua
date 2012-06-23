----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (60,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-plot', false, 'plot training and test errors, live')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | SGD+LBFGS')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 3, 'start averaging (ASGD) or switch methods (SGD+LFBGS) at the t0-th epoch')
cmd:option('-maxIter', 3, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 1, 'nb of threads')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialSubtractiveNormalization(1, image.gaussian1D(15)))
      model:add(nn.SpatialConvolution(1, 16, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialSubtractiveNormalization(16, image.gaussian1D(15)))
      model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 128, 4), 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(128*5*5))
      model:add(nn.Linear(128*5*5, 200))
      model:add(nn.Tanh())
      model:add(nn.Linear(200,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
--criterion = nn.DistKLDivCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
mean, std = trainData:normalizeGlobal()

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- display function
function display(input)
   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter%10 == 0 then
      if opt.model == 'convnet' then
         win_w1 = image.display{image=model:get(2).weight, zoom=4, nrow=10,
                                min=-1, max=1,
                                win=win_w1, legend='stage 1: weights', padding=1}
         win_w2 = image.display{image=model:get(6).weight, zoom=4, nrow=30,
                                min=-1, max=1,
                                win=win_w2, legend='stage 2: weights', padding=1}
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{image=W1, zoom=0.5,
                                min=-1, max=1,
                                win=win_w1, legend='W1 weights'}
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{image=W2, zoom=0.5,
                                min=-1, max=1,
                                win=win_w2, legend='W2 weights'}
      end
   end
   iter = iter + 1
end

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- if SGD+LBFGS, change batch size when going from SGD to LBFGS
   if opt.optimization == 'SGD+LBFGS' then
      _batchSize = _batchSize or opt.batchSize
      if epoch < opt.t0 then
         opt.batchSize = 1  -- SGD
      else
         opt.batchSize = _batchSize -- final batch size
      end
      print('<trainer> setting batch size to ' .. opt.batchSize)
   end

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])

                          -- visualize?
                          if opt.visualize then
                             display(inputs[i])
                          end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {maxIter = opt.maxIter,
                             lineSearch = optim.lswolfe}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'SGD+LBFGS' then
         if epoch < opt.t0 then
            config = config or {learningRate = opt.learningRate,
                                weightDecay = opt.weightDecay,
                                momentum = opt.momentum,
                                learningRateDecay = 5e-7}
            optim.sgd(feval, parameters, config)
         else
            config2 = config2 or {maxIter = opt.maxIter,
                                  lineSearch = optim.lswolfe}
            optim.lbfgs(feval, parameters, config2)
         end

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * (opt.t0-1)}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if sys.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local sample = dataset[t]
      local input = sample[1]
      local _,target = sample[2]:max(1)
      target = target:squeeze()

      -- test sample
      confusion:add(model:forward(input), target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   train(trainData)
   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end
