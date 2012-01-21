----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. It illustrates several 
-- points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset, from a simple directory of PNGs
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'image'
require 'nnx'
require 'optim'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
op = xlua.OptionParser('%prog [options]')
op:option{'-s', '--save', action='store', dest='save',
          default=fname:gsub('.lua','') .. '/digit.net',
          help='file to save network after each epoch'}
op:option{'-l', '--load', action='store', dest='network',
          help='reload pretrained network'}

op:option{'-m', '--model', action='store', dest='model',
          help='model type: convnet | mlp | linear', default='convnet'}

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

op:option{'-op', '--optimization', action='store', dest='optimization',
          default='SGD',
          help='optimization method: SGD | ASGD | CG'}
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
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

if not opt.network then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialSubtractiveNormalization(1, image.gaussian1D(15)))
      model:add(nn.SpatialConvolution(1, 50, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialSubtractiveNormalization(50, image.gaussian1D(15)))
      model:add(nn.SpatialConvolutionMap(nn.tables.random(50, 128, 10), 5, 5))
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
      op:help()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
criterion = nn.DistNLLCriterion()
criterion.targetIsProbability = true

----------------------------------------------------------------------
-- get/create dataset
--
path_dataset = opt.dataset

-- retrieve dataset from the web
if not sys.dirp(path_dataset) then
   local path = sys.dirname(path_dataset)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p ' .. path .. '; '..
              'cd ' .. path .. '; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
end

-- train on full dataset, or smaller toy subset
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag --full to use 60000 samples)')
end

-- create training set
trainData = nn.DataList()
for i,class in ipairs(classes) do
   local dir = sys.concat(path_dataset,'train',class)
   local subset = nn.DataSet{dataSetFolder = dir,
                             cacheFile = sys.concat(path_dataset,'train',class..'-cache'),
                             nbSamplesRequired = nbTrainingPatches/10, channels=1}
   subset:shuffle()
   trainData:appendDataSet(subset, class)
end

-- create test set
testData = nn.DataList()
for i,class in ipairs(classes) do
   local subset = nn.DataSet{dataSetFolder = sys.concat(path_dataset,'test',class),
                             cacheFile = sys.concat(path_dataset,'test',class..'-cache'),
                             nbSamplesRequired = nbTestingPatches/10, channels=1}
   subset:shuffle()
   testData:appendDataSet(subset, class)
end

-- set both datasets to generate valid probability distributions
-- as target vectors
trainData.targetIsProbability = true
testData.targetIsProbability = true

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = nn.ConfusionMatrix(classes)

-- log results to files
trainLogger = nn.Logger(sys.dirname(opt.save) .. '/train.log')
testLogger = nn.Logger(sys.dirname(opt.save) .. '/test.log')

-- display function
function display(input)
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if opt.model == 'convnet' then
      win_w1 = image.display{image=model:get(2).weight, zoom=4, nrow=10,
                             win=win_w1, legend='stage 1: weights', padding=1}
      win_w2 = image.display{image=model:get(6).weight, zoom=4, nrow=30,
                             win=win_w2, legend='stage 2: weights', padding=1}
      win_g1 = image.display{image=model:get(2).gradWeight, zoom=4, nrow=10,
                             win=win_g1, legend='stage 1: gradients', padding=1}
      win_g2 = image.display{image=model:get(6).gradWeight, zoom=4, nrow=30,
                             win=win_g2, legend='stage 2: gradients', padding=1}
   elseif opt.model == 'mlp' then
      local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
      win_w1 = image.display{image=W1, zoom=0.5,
                              win=win_w1, legend='W1 weights'}
      local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
      win_w2 = image.display{image=W2, zoom=0.5,
                              win=win_w2, legend='W2 weights'}
   end
end

-- training function
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
         config = config or {length = opt.maxIteration}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = 1e-2,
                             weightDecay = 1e-3,
                             momentum = 0,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = 1e-2,
                             t0 = nbTrainingPatches}
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
      torch.save(filename, model)
   end

   -- next epoch
   epoch = epoch + 1
end

-- test function
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
   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   trainLogger:plot()
   testLogger:plot()
end
