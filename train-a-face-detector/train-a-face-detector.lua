----------------------------------------------------------------------
-- A simple script that trains a conv net on a face detection dataset,
-- using stochastic gradient descent.
--
-- This script demonstrates a classical example of training a simple
-- convolutional network on a binary classification problem. It
-- illustrates several points:
-- 1/ description of the network
-- 2/ choice of a cost function (criterion) to minimize
-- 3/ instantiation of a trainer, with definition of learning rate, 
--    decays, and momentums
-- 4/ creation of a dataset, from multiple directories of PNGs
-- 5/ running the trainer, which consists in showing all PNGs+Labels
--    to the network, and performing stochastic gradient descent 
--    updates
--
-- Clement Farabet, Benoit Corda  |  July  7, 2011, 12:45PM
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
          default=fname:gsub('.lua','') .. '/face.net',
          help='file to save network after each epoch'}
op:option{'-l', '--load', action='store', dest='network',
          help='reload pretrained network'}
op:option{'-d', '--dataset', action='store', dest='dataset', 
          default='../datasets/faces_cut_yuv_32x32/',
          help='path to MNIST root dir'}
op:option{'-w', '--www', action='store', dest='www', 
          default='http://data.neuflow.org/data/faces_cut_yuv_32x32.tar.gz',
          help='path to retrieve dataset online (if not available locally)'}
op:option{'-t', '--testset', action='store', dest='ratio', 
          help='percentage of samples to use for testing', default=0.2}
op:option{'-p', '--patches', action='store', dest='patches', default='all',
          help='nb of patches to use'}
op:option{'-v', '--visualize', action='store_true', dest='visualize',
          help='visualize the datasets'}
op:option{'-sd', '--seed', action='store', dest='seed', default=0,
          help='use fixed seed for randomized initialization'}
opt = op:parse()

torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.seed)

----------------------------------------------------------------------
-- define network to train: CSCF
--
if not opt.network then
   model = nn.Sequential()
   model:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(5)))
   model:add(nn.SpatialConvolution(1, 8, 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(4, 4, 4, 4))
   model:add(nn.SpatialConvolutionMap(nn.tables.random(8, 64, 4), 7, 7))
   model:add(nn.Tanh())
   model:add(nn.Reshape(64))
   model:add(nn.Linear(64,2))
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

----------------------------------------------------------------------
-- training criterion: a simple Mean-Square Error
--
criterion = nn.MSECriterion()
criterion.sizeAverage = true

----------------------------------------------------------------------
-- create dataset
--
if not sys.dirp(opt.dataset) then
   local path = sys.dirname(opt.dataset)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p ' .. path .. '; '..
              'cd ' .. path .. '; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
end

if opt.patches ~= 'all' then
   opt.patches = math.floor(opt.patches/3)
end

-- Faces:
dataFace = nn.DataSet{dataSetFolder=opt.dataset..'face', 
                      cacheFile=opt.dataset..'face',
                      nbSamplesRequired=opt.patches,
                      channels=1}
dataFace:shuffle()

-- Backgrounds:
dataBG = nn.DataSet{dataSetFolder=opt.dataset..'bg',
                    cacheFile=opt.dataset..'bg',
                    nbSamplesRequired=opt.patches,
                    channels=1}
dataBGext = nn.DataSet{dataSetFolder=opt.dataset..'bg-false-pos-interior-scene',
                       cacheFile=opt.dataset..'bg-false-pos-interior-scene',
                       nbSamplesRequired=opt.patches,
                       channels=1}
dataBG:appendDataSet(dataBGext)
dataBG:shuffle()

-- pop subset for testing
testFace = dataFace:popSubset{ratio=opt.ratio}
testBg = dataBG:popSubset{ratio=opt.ratio}

-- training set
trainData = nn.DataList()
trainData:appendDataSet(dataFace,'Faces')
trainData:appendDataSet(dataBG,'Background')

-- testing set
testData = nn.DataList()
testData:appendDataSet(testFace,'Faces')
testData:appendDataSet(testBg,'Background')

-- display
if opt.visualize then
   trainData:display(100,'trainData')
   testData:display(100,'testData')
end

----------------------------------------------------------------------
-- train/test
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix{'Face','Background'}

-- log results to files
trainLogger = optim.Logger(paths.concat(sys.dirname(opt.save), 'train.log'))
testLogger = optim.Logger(paths.concat(sys.dirname(opt.save), 'test.log'))

-- optim config
config = {learningRate = 1e-3, weightDecay = 1e-3,
          momentum = 0.1, learningRateDecay = 5e-7}

batchSize = 1
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,dataset:size(),batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]
         local target = sample[2]
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
      optim.sgd(feval, parameters, config)
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
   local filename = opt.save
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
