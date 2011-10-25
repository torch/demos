--------------------------------------------------------------------------------
-- A simple script that trains a ConvNet on a face dataset, using 
-- stochastic gradient descent.
-- 
-- TYPE:
-- $$ qlua train-on-faces.lua --help
--
-- Authors: Corda / Farabet
--------------------------------------------------------------------------------
require 'XLearn'

-- Option parsing
op = OptionParser('%prog [options]')
op:add_option{'-n', '--network', action='store', dest='network', 
              help='path to existing [trained] network'}
op:add_option{'-d', '--dataset', action='store', dest='dataset', 
              help='path to dataset'}
op:add_option{'-t', '--testset', action='store', dest='ratio', 
              help='percentage of samples to use for testing', default=0.2}
op:add_option{'-p', '--patches', action='store', dest='patches', 
              help='nb of patches to use'}
op:add_option{'-z', '--normalize', action='store', dest='norm', 
              help='type of norm to use: regular | fixedThres | neuFlow', default='regular'}
op:add_option{'-s', '--show', action='store', dest='show', 
              help='show dataset', default=false}
op:add_option{'-r', '--randseed', action='store', dest='seed',
              help='force random seed (if not provided, then initial conditions are random)'}
options,args = op:parse_args()

-- make local scratch dir
os.execute('mkdir -p scratch')

-- use seed (for repeatable experiments)
if options.seed then
   random.manualSeed(options.seed)
end

-- First pass: create network
if not options.network then
   -- localnorm
   if options.norm == 'regular' then
      localnorm = nn.LocalNorm(image.gaussian{width=7, amplitute=1}, 1)
   elseif options.norm == 'fixedThres' then
      localnorm = nn.LocalNorm(image.gaussian{width=7, amplitute=1}, 1, 2/256)
   elseif options.norm == 'neuFlow' then
      localnorm = nn.LocalNorm_hardware(image.gaussian{width=7, amplitute=1}, 1, 2/256, true)
   end

   -- Build network
   convnet = nn.Sequential()
   convnet:add(localnorm)
   convnet:add(nn.SpatialConvolution(1, 8, 5, 5))
   convnet:add(nn.Tanh())
   convnet:add(nn.AbsModule())
   convnet:add(nn.SpatialSubSampling(8, 4, 4, 4, 4))
   convnet:add(nn.Tanh())
   convnet:add(nn.SpatialConvolutionTable(nn.SpatialConvolutionTable:KorayTable(8,20,4), 7, 7))
   convnet:add(nn.Tanh())
   convnet:add(nn.SpatialLinear(20,2))
else
   -- reload trained network
   print('# reloading previously trained network')
   convnet = nn.Sequential():readf(options.network)
end

-- criterion and trainer
trainer = nn.StochasticTrainer(convnet, nn.MSECriterion())
trainer:setShuffle(false)
trainer.learningRate = 0.001
trainer.learningRateDecay = 0.01
trainer.maxEpoch = 50

-- Datasets path
datasetPath = options.dataset or '../datasets/faces_cut_yuv_32x32/'

-- nb of patches
if options.patches then
   nbpatches = math.floor(options.patches/3)
else
   nbpatches = 'all'
end

-- Faces:
dataFace = DataSet()
dataFace:load{dataSetFolder=datasetPath..'face', 
              cacheFile=datasetPath..'face',
              nbSamplesRequired=nbpatches,
              chanels=1}
dataFace:shuffle()

-- Backgrounds:
dataBG = DataSet{dataSetFolder=datasetPath..'bg',
                 cacheFile=datasetPath..'bg',
                 nbSamplesRequired=nbpatches,
                 chanels=1}
dataBGext = DataSet{dataSetFolder=datasetPath..'bg-false-pos-interior-scene',
                    cacheFile=datasetPath..'bg-false-pos-interior-scene',
                    nbSamplesRequired=nbpatches,
                    chanels=1}
dataBG:appendDataSet(dataBGext)
dataBG:shuffle()

-- pop subset for testing
ratio = options.ratio
testFace = dataFace:popSubset{ratio=ratio}
testBg = dataBG:popSubset{ratio=ratio}

-- training set
trainData = DataList()
trainData:appendDataSet(dataFace,'Faces')
trainData:appendDataSet(dataBG,'Bg')

-- testing set
testData = DataList()
testData:appendDataSet(testFace,'Faces')
testData:appendDataSet(testBg,'Bg')

-- display sets
if options.show then
   trainData:display{nbSamples=300, title='Training Set', gui=false}
   testData:display{nbSamples=300, title='Test Set', gui=false}
end

-- training hooks
confusion = nn.ConfusionMatrix(2)

trainer.hookTrainSample = function(trainer, sample)
   -- update confusion matrix
   confusion:add(trainer.module.output[1][1], sample[2][1][1])
end

trainer.hookTestSample = function(trainer, sample)
   -- update confusion matrix
   confusion:add(trainer.module.output[1][1], sample[2][1][1])
end

trainer.hookTrainEpoch = function(trainer)
   -- print confusion
   print(confusion)
   confusion:zero()

   -- run on test_set
   trainer:test(testData)

   -- print confusion
   print(confusion)
   confusion:zero()

   -- save net
   local filename = paths.concat('scratch', (options.saveto or 'network-faces')..'-'..os.date("%Y_%m_%d@%X"))
   print('# saving network to '..filename)
   trainer.module:writef(filename)
end

-- Run trainer
trainer:train(trainData)
