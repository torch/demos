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
xrequire ('image', true)
xrequire ('nnx', true)

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
          help='optimization method: SGD or BFGS'}
op:option{'-bs', '--batchSize', action='store', dest='batchSize',
          default=1,
          help='mini-batch size'}
op:option{'-mi', '--maxIteration', action='store', dest='bfgsMaxIteration',
          default=20,
          help='maximum nb of iterations for each mini-batch'}
op:option{'-me', '--maxEval', action='store', dest='maxEval',
          default=0,
          help='maximum nb of function evaluations for each mini-batch'}
op:option{'-pz', '--parallelize', action='store', dest='parallelize',
          default=1,
          help='parallelize mini-batch computations onto N cores'}

opt = op:parse()

torch.setdefaulttensortype('torch.DoubleTensor')

if opt.seed then
   random.manualSeed(opt.seed)
end

----------------------------------------------------------------------
-- define network to train: CSCSCF
--

nbClasses = 10
connex = {12,32,128}
fanin = {1,8,32}

if not opt.network then
   convnet = nn.Sequential()
   convnet:add(nn.SpatialConvolution(1,connex[1], 5, 5))
   convnet:add(nn.Tanh())
   convnet:add(nn.SpatialSubSampling(connex[1], 2, 2, 2, 2))
   convnet:add(nn.Tanh())
   convnet:add(nn.SpatialConvolution(connex[1],connex[2], 5, 5))
   convnet:add(nn.Tanh())
   convnet:add(nn.SpatialSubSampling(connex[2], 2, 2, 2, 2))
   convnet:add(nn.Tanh())
   convnet:add(nn.SpatialConvolution(connex[2],connex[3], 5, 5))
   convnet:add(nn.Tanh())
   convnet:add(nn.Reshape(connex[3]))
   convnet:add(nn.Linear(connex[3],nbClasses))
else
   print('<trainer> reloading previously trained network')
   convnet = nn.Sequential()
   convnet:read(torch.DiskFile(opt.network))
end

----------------------------------------------------------------------
-- training criterion: a simple Mean-Square Error
--
if opt.error == 'mse' then
   criterion = nn.MSECriterion()
   criterion.sizeAverage = true
elseif opt.error == 'nll' then
   criterion = nn.DistNLLCriterion()
   criterion.targetIsProbability = true
end

----------------------------------------------------------------------
-- trainer: std stochastic trainer, plus training hooks
--
if opt.optimization == 'BFGS' then
   optimizer = nn.LBFGSOptimization{module = convnet,
                                    criterion = criterion,
                                    parallelize = tonumber(opt.parallelize),
				    maxEvaluation = opt.maxEval,
				    maxIterations = opt.bfgsMaxIteration,
                                    verbose = 2}
   dispProgress = false
else
   optimizer = nn.SGDOptimization{module = convnet,
                                  criterion = criterion,
                                  parallelize = tonumber(opt.parallelize),
                                  learningRate = 1e-2,
                                  weightDecay = 1e-4,
                                  learningRateDecay = 5e-7,
                                  momentum = 0.5}
   dispProgress = true
end

batchSize = opt.batchSize

trainer = nn.OnlineTrainer{module = convnet, 
                           criterion = criterion,
                           optimizer = optimizer,
                           maxEpoch = 500,
                           dispProgress = dispProgress,
                           batchSize = batchSize,
                           save = opt.save}

classes = {'1','2','3','4','5','6','7','8','9','10'}

confusion = nn.ConfusionMatrix(classes)

trainLogger = nn.Logger(sys.dirname(opt.save) .. '/train.log')
testLogger = nn.Logger(sys.dirname(opt.save) .. '/test.log')

optimizer.posthook = function(optimizer, sample)
   confusion:add(optimizer.module.output, sample[2])
end

trainer.hookTestSample = function(trainer, sample)
   confusion:add(trainer.module.output, sample[2])
end

trainer.hookTrainEpoch = function(trainer)
   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- run on test_set
   trainer:test(testData)

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- plot errors
   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   trainLogger:plot()
   testLogger:plot()
end

----------------------------------------------------------------------
-- get/create dataset
--
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
-- and train !!
--
trainer:train(trainData)
