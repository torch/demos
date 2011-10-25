----------------------------------------------------------------------
-- A simple script that trains a ConvNet on the MNIST dataset,
-- using stochastic gradient descent.
--
-- C.Farabet
----------------------------------------------------------------------

require 'XLearn'


----------------------------------------------------------------------
-- parse options
--
op = OptionParser('%prog [options]')
op:add_option{'-l', '--load', action='store', dest='network',
              help='path to existing [trained] network'}
op:add_option{'-s', '--save', action='store', dest='saveto',
              help='file name to save network [saving is done after each epoch]'}
op:add_option{'-d', '--dataset', action='store', dest='dataset',
              help='path to dataset'}
op:add_option{'-n', '--show', action='store', dest='nb_samples',
              help='show N samples from dataset'}
op:add_option{'-f', '--full', action='store_true', dest='full',
              help='use full dataset (60,000 samples) to train'}
op:add_option{'-r', '--randseed', action='store', dest='seed',
              help='force random seed (if not provided, then initial conditions are random)'}

options,args = op:parse_args()


----------------------------------------------------------------------
-- To save networks
--
os.execute('mkdir -p scratch')


----------------------------------------------------------------------
-- ConvNet to train: CSCSCF
--

local nbClasses = 10
local connex = {6,16,120}
local fanin = {1,6,16}

-- use seed (for repeatable experiments)
if options.seed then
   random.manualSeed(options.seed)
end

-- Build network
convnet = nn.SequentialHessian()
convnet:add(nn.SpatialConvolutionTableHessian(nn.SpatialConvolutionTable:FullTable(1,connex[1]), 5, 5))
convnet:add(nn.TanhHessian())
convnet:add(nn.SpatialSubSamplingHessian(connex[1], 2, 2, 2, 2))
convnet:add(nn.TanhHessian())
convnet:add(nn.SpatialConvolutionTableHessian(nn.SpatialConvolutionTable:FullTable(connex[1],connex[2]), 5, 5))
convnet:add(nn.TanhHessian())
convnet:add(nn.SpatialSubSamplingHessian(connex[2], 2, 2, 2, 2))
convnet:add(nn.TanhHessian())
convnet:add(nn.SpatialConvolutionTableHessian(nn.SpatialConvolutionTable:FullTable(connex[2],connex[3]), 5, 5))
convnet:add(nn.TanhHessian())
convnet:add(nn.SpatialLinearHessian(connex[3],nbClasses))


----------------------------------------------------------------------
-- learning criterion: simple Mean-Square error
--
criterion = nn.MSECriterionHessian()
criterion.sizeAverage = true


----------------------------------------------------------------------
-- trainer: std stochastic trainer
--
trainer = nn.StochasticHessianTrainer(convnet, criterion)
trainer:setShuffle(false)
trainer.hessianUpdateFrequency = 500
trainer.boundOnLearningRates = 0.1
trainer.hessianSamplesUsed = 100
trainer.learningRate = 1e-2
trainer.learningRateDecay = 0
trainer.weightDecay = 1e-5
trainer.maxEpoch = 50


----------------------------------------------------------------------
-- load datasets
--
path_dataset = options.dataset or '../datasets/mnist/'
path_trainData = paths.concat(path_dataset,'train-images-idx3-ubyte')
path_trainLabels = paths.concat(path_dataset,'train-labels-idx1-ubyte')
path_testData = paths.concat(path_dataset,'t10k-images-idx3-ubyte')
path_testLabels = paths.concat(path_dataset,'t10k-labels-idx1-ubyte')

trainData = {}
testData = {}

nbTrainingPatches = 2000
nbTestingPatches = 1000

if options.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000 
else
   print('# warning: only using 2000 samples to train quickly (use flag --full to use 60000 samples)')
end

-- load data+labels
local data = toolBox.loadIDX(path_trainData):resize(28,28,nbTrainingPatches)
local labels = toolBox.loadIDX(path_trainLabels):resize(nbTrainingPatches)
for i=1,data:size(3) do
   local target = torch.Tensor(1,1,nbClasses):fill(-1)
   target[1][1][labels[i]+1] = 1
   local sample = torch.Tensor(32,32,1):fill(0)
   sample:narrow(1,3,28):narrow(2,3,28):copy(data:narrow(3,i,1)):mul(0.01)
   trainData[i] = {sample,target}
end
trainData.size = function (self) return #self end

-- load data+labels
data = toolBox.loadIDX(path_testData):resize(28,28,nbTestingPatches)
labels = toolBox.loadIDX(path_testLabels):resize(nbTestingPatches)
for i=1,data:size(3) do
   local target = torch.Tensor(1,1,nbClasses):fill(-1)
   target[1][1][labels[i]+1] = 1
   local sample = torch.Tensor(32,32,1):fill(0)
   sample:narrow(1,3,28):narrow(2,3,28):copy(data:narrow(3,i,1)):mul(0.01)
   testData[i] = {sample,target}
end
testData.size = function (self) return #self end

-- display ?
if options.nb_samples then
   local samples = {}
   for i = 1,options.nb_samples do
      table.insert(samples, trainData[i][1])
   end
   image.displayList{images=samples, gui=false}
end


----------------------------------------------------------------------
-- training hooks
--
confusion = nn.ConfusionMatrix(nbClasses)

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
   local filename = paths.concat('scratch', (options.saveto or 'network-mnist')..'-'..os.date("%Y_%m_%d@%X"))
   print('# saving network to '..filename)
   trainer.module:writef(filename)
end


----------------------------------------------------------------------
-- run trainer
--
trainer:train(trainData)
