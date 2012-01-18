----------------------------------------------------------------------
-- A simple script that trains a conv net on the MNIST dataset,
-- using stochastic gradient descent.
--
-- This script demonstrates a classical example of training a simple
-- convolutional network on a 10-class classification problem. It
-- illustrates several points:
-- 1/ description of the network
-- 2/ choice of a cost function (criterion) to minimize
-- 3/ instantiation of a trainer, with definition of learning rate,
--    decays, and momentums
-- 4/ creation of a dataset, from a simple directory of PNGs
-- 5/ running the trainer, which consists in showing all PNGs+Labels
--    to the network, and performing stochastic gradient descent
--    updates
--
-- Clement Farabet  |  July  7, 2011, 12:44PM
----------------------------------------------------------------------

require 'xlua'
require 'image'
require 'nnx'
require 'optim'

----------------------------------------------------------------------
-- parse options
--
dname,fname = sys.fpath()
op = xlua.OptionParser('%prog [options]')
op:option{'-s', '--save', action='store', dest='save',
          default=fname:gsub('.lua','') .. '/digit.net',
          help='file to save network after each epoch'}
op:option{'-l', '--load', action='store', dest='network',
          help='reload pretrained network'}
op:option{'-d', '--dataset', action='store', dest='dataset',
          default='../datasets/mnist',
          help='path to MNIST root dir'}
op:option{'-w', '--www', action='store', dest='www',
          default='http://data.neuflow.org/data/mnist.tgz',
          help='path to retrieve dataset online (if not available locally)'}

op:option{'-f', '--full', action='store_true', dest='full',
          help='use full dataset (60,000 samples) to train'}

op:option{'-v', '--visualize', action='store_true', dest='visualize',
          help='visualize the datasets'}

op:option{'-sd', '--seed', action='store', dest='seed',
          help='use fixed seed for randomized initialization'}

op:option{'-ls', '--loss', action='store', dest='error',
          help='type of loss function: mse OR nll', default='nll'}

op:option{'-op', '--optimization', action='store', dest='optimization',
          default='SGD',
          help='optimization method: SGD | CG'}
op:option{'-bs', '--batchSize', action='store', dest='batchSize',
          default=1,
          help='mini-batch size'}
op:option{'-mi', '--maxIteration', action='store', dest='maxIteration',
          default=5,
          help='maximum nb of iterations for each mini-batch'}

opt = op:parse()

torch.setdefaulttensortype('torch.DoubleTensor')

if opt.seed then
   random.manualSeed(opt.seed)
end

----------------------------------------------------------------------
-- define network to train: CSCSCF
--

nbClasses = 10
connex = {50,128,200}
fanin = {-1,10,-1}

if not opt.network then
   module = nn.Sequential()
   module:add(nn.SpatialConvolution(1, connex[1], 5, 5))
   module:add(nn.Tanh())
   module:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   module:add(nn.SpatialConvolutionMap(nn.tables.random(connex[1], connex[2], fanin[2]), 5, 5))
   module:add(nn.Tanh())
   module:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   module:add(nn.SpatialConvolution(connex[2], connex[3], 5, 5))
   module:add(nn.Tanh())

   module:add(nn.Reshape(connex[3]))
   module:add(nn.Linear(connex[3],nbClasses))
else
   print('<trainer> reloading previously trained network')
   module = nn.Sequential()
   module:read(torch.DiskFile(opt.network))
end

parameters = nnx.flattenParameters(nnx.getParameters(module))
gradParameters = nnx.flattenParameters(nnx.getGradParameters(module))

----------------------------------------------------------------------
-- training criterion: MSE or Cross-entropy
--
if opt.error == 'mse' then
   criterion = nn.MSECriterion()
   criterion.sizeAverage = true
elseif opt.error == 'nll' then
   criterion = nn.DistNLLCriterion()
   criterion.targetIsProbability = true
end

----------------------------------------------------------------------
-- get/create dataset
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

path_dataset = opt.dataset
if not sys.dirp(path_dataset) then
   local path = sys.dirname(path_dataset)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p ' .. path .. '; '..
              'cd ' .. path .. '; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
end

if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag --full to use 60000 samples)')
end

trainData = nn.DataList()
for i,class in ipairs(classes) do
   local dir = sys.concat(path_dataset,'train',class)
   local subset = nn.DataSet{dataSetFolder = dir,
                             cacheFile = sys.concat(path_dataset,'train',class..'-cache'),
                             nbSamplesRequired = nbTrainingPatches/10, channels=1}
   subset:shuffle()
   trainData:appendDataSet(subset, class)
end

testData = nn.DataList()
for i,class in ipairs(classes) do
   local subset = nn.DataSet{dataSetFolder = sys.concat(path_dataset,'test',class),
                             cacheFile = sys.concat(path_dataset,'test',class..'-cache'),
                             nbSamplesRequired = nbTestingPatches/10, channels=1}
   subset:shuffle()
   testData:appendDataSet(subset, class)
end

if opt.error == 'nll' then
   trainData.targetIsProbability = true
   testData.targetIsProbability = true
end

if opt.visualize then
   trainData:display(100,'trainData')
   testData:display(100,'testData')
end

----------------------------------------------------------------------
-- define training and testing functions
--

confusion = nn.ConfusionMatrix(classes)

trainLogger = nn.Logger(sys.dirname(opt.save) .. '/train.log')
testLogger = nn.Logger(sys.dirname(opt.save) .. '/test.log')

function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

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
         local input = sample[1]
         local target = sample[2]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- creature closure to evaluate f(X) and df/dX
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
                          local output = module:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          module:backward(inputs[i], df_do)

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
         config = config or {length = opt.maxIteration}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = 1e-2,
                             weightDecay = 1e-5,
                             momentum = 0,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = 1e-2,
                             t0 = nbTrainingPatches * 1}
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
   if opt.save then 
      -- save network
      local filename = opt.save
      os.execute('mkdir -p ' .. sys.dirname(filename))
      if sys.filep(filename) then
         os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
      end
      print('<trainer> saving network to '..filename)
      torch.save(filename, module)
   end

   -- next epoch
   epoch = epoch + 1
end

function test(dataset)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      print('averaging')
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
      local target = sample[2]

      -- test sample
      confusion:add(module:forward(input), target)
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
   trainLogger:plot()
   testLogger:plot()
end
