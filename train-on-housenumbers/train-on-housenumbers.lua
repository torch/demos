----------------------------------------------------------------------
-- This script shows how to train different models on the street
-- view house number dataset,
-- using multiple optimization techniques (SGD, ASGD, CG)
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
-- Note: the architecture of the convnet is based on Pierre Sermanet's
-- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
-- the use of LP-pooling (with P=2) has a very positive impact on
-- generalization. Normalization is not done exactly as proposed in
-- the paper, and low-level (first layer) features are not fed to
-- the classifier.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('HouseNumber Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-extra', false, 'use extra training samples dataset (~500,000 extra training samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','0'}

if opt.network == '' then
   ------------------------------------------------------------
   -- convolutional network 
   -- this is a typical convolutional network for vision:
   --   1/ the image is transformed into Y-UV space
   --   2/ the Y (luminance) channel is locally normalized,
   --      while the U/V channels are more loosely normalized
   --   3/ the first two stages features are locally pooled
   --      using LP-pooling (P=2)
   --   4/ a two-layer neural network is applied on the
   --      representation
   ------------------------------------------------------------
   -- top container
   model = nn.Sequential()
   -- stage 1 : filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(16,2,2,2,2,2))
   -- stage 2 : filter bank -> squashing -> max pooling
   model:add(nn.SpatialSubtractiveNormalization(16, image.gaussian1D(7)))
   model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 256, 4), 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(256,2,2,2,2,2))
   -- stage 3 : standard 2-layer neural network
   model:add(nn.SpatialSubtractiveNormalization(256, image.gaussian1D(7)))
   model:add(nn.Reshape(256*5*5))
   model:add(nn.Linear(256*5*5, 128))
   model:add(nn.Tanh())
   model:add(nn.Linear(128,#classes))
   model:add(nn.LogSoftMax())
   ------------------------------------------------------------
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<trainer> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.extra then
   trsize = 73257 + 531131
   tesize = 26032
else
   print '<trainer> WARNING: using reduced train set'
   print '(use -extra to use complete training set, with extra samples)'
   trsize = 73257
   tesize = 26032
end

www = 'http://data.neuflow.org/data/housenumbers/'
train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'
extra_file = 'extra_32x32.t7'
if not paths.filep(train_file) then
   os.execute('wget ' .. www .. train_file)
end
if not paths.filep(test_file) then
   os.execute('wget ' .. www .. test_file)
end
if opt.extra and not paths.filep(extra_file) then
   os.execute('wget ' .. www .. extra_file)   
end

loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

if opt.extra then
   loaded = torch.load(extra_file,'ascii')
   trdata = torch.Tensor(trsize,3,32,32)
   trdata[{ {1,(#trainData.data)[1]} }] = trainData.data
   trdata[{ {(#trainData.data)[1]+1,-1} }] = loaded.X:transpose(3,4)
   trlabels = torch.Tensor(trsize)
   trlabels[{ {1,(#trainData.labels)[1]} }] = trainData.labels
   trlabels[{ {(#trainData.labels)[1]+1,-1} }] = loaded.y[1]
   trainData = {
      data = trdata,
      labels = trlabels,
      size = function() return trsize end
   }
end

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'

-- preprocess requires floating point
trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7)):float()
for i = 1,trainData:size() do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end
-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- display
if opt.visualize then
   require 'image'
   local trset = trainData.data[{ {1,100} }]
   local teset = testData.data[{ {1,100} }]
   image.display{image=trset, legend='training set', nrow=10, padding=1}
   image.display{image=teset, legend='test set', nrow=10, padding=1}
end

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

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
         local input = dataset.data[shuffle[i]]:double()
         local target = dataset.labels[shuffle[i]]
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
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = trsize * opt.t0}
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
   local filename = paths.concat(opt.save, 'house.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
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
      local input = dataset.data[t]:double()
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
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
   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   if opt.plot then
      trainLogger:plot()
      testLogger:plot()
   end
end
