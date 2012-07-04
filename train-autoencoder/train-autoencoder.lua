----------------------------------------------------------------------
-- This script shows how to train autoencoders on natural images,
-- using the unsup package.
--
-- Borrowed from Koray Kavukcuoglu's unsup repo
----------------------------------------------------------------------

require 'unsup'
require 'image'
require 'gnuplot'

require 'autoencoder-data'
if not arg then arg = {} end

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
cmd:option('-dir','outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 123211, 'initial random seed')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 32, 'number of output convolutional filters')
cmd:option('-kernelsize', 9, 'size of convolutional kernels')
cmd:option('-inputsize', 9, 'size of each input patch')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-eta', 0.01, 'learning rate')
cmd:option('-eta_encoder', 0, 'encoder learning rate')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-decay', 0, 'weigth decay')
cmd:option('-maxiter', 1000000, 'max number of updates')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', false, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:option('-conv', false, 'force convolutional dictionary')
cmd:option('-threads', 4, 'threads')
cmd:text()

params = cmd:parse(arg)

rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)
cmd:log(params.rundir .. '/log', params)

torch.manualSeed(params.seed)

torch.setnumthreads(params.threads)

----------------------------------------------------------------------
-- load data
--
filename = paths.basename(params.datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. params.datafile .. '; '.. 'tar xvf ' .. filename)
end
dataset = getdata(filename, params.inputsize)

if params.display then
   displayData(dataset, 400, 20, 2)
end

----------------------------------------------------------------------
-- create model
--
if params.inputsize == params.kernelsize and params.conv == false then

   -- params
   inputSize = params.inputsize*params.inputsize
   outputSize = params.nfiltersout

   -- encoder
   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputSize,outputSize))

   -- decoder is L1 solution
   decoder = unsup.LinearFistaL1(inputSize, outputSize, params.lambda)

   -- PSD autoencoder
   module = unsup.PSD(encoder, decoder, params.beta)

else

   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputsize, params.inputsize

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- decoder is L1 solution:
   decoder = unsup.SpatialConvFistaL1(decodertable, kw, kh, iw, ih, params.lambda)

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))

   -- PSD autoencoder
   module = unsup.PSD(encoder, decoder, params.beta)

   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()
end

----------------------------------------------------------------------
-- training parameters
--
-- learning rates
if params.eta_encoder == 0 then params.eta_encoder = params.eta end
params.eta = torch.Tensor({params.eta_encoder, params.eta})

-- do learrning rate hacks
kex.nnhacks()

----------------------------------------------------------------------
-- train model
--
local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local currentLearningRate = params.eta

local err = 0
local iter = 0
for t = 1,params.maxiter do

   -- get next sample
   local example = dataset[t]

   -- update model on sample (one gradient descent step)
   local updateSample = function()
      local input = example[1]
      local target = example[2]
      local err,h = module:updateOutput(input, target)
      module:zeroGradParameters()
      module:updateGradInput(input, target)
      module:accGradParameters(input, target)
      module:updateParameters(currentLearningRate)
      return err, #h
   end
   local serr, siter = updateSample()
   err = err + serr
   iter = iter + siter

   -- gather/print statistics
   if math.fmod(t , params.statinterval) == 0 then
      -- training error / iteration
      avTrainingError[t/params.statinterval] = err/params.statinterval
      avFistaIterations[t/params.statinterval] = iter/params.statinterval

      -- report
      print('# iter=' .. t .. ' eta = ( ' .. currentLearningRate[1] .. 
            ', ' .. currentLearningRate[2] .. ' ) current error = ' .. err)

      -- plot filters
      if params.conv then
         de = image.toDisplayTensor{input=module.encoder.modules[1].weight,padding=2,nrow=8,symmetric=true}
         dd = image.toDisplayTensor{input=module.decoder.D.weight,padding=2,nrow=8,symmetric=true}
      else
         dd = image.toDisplayTensor{input=module.decoder.D.weight:transpose(1,2):unfold(2,params.inputsize,params.inputsize),padding=2,nrow=8,symmetric=true}
         de = image.toDisplayTensor{input=module.encoder.modules[1].weight:unfold(2,params.inputsize,params.inputsize),padding=2,nrow=8,symmetric=true}
      end
      image.save(params.rundir .. '/filters_dec_' .. t .. '.jpg', dd)
      image.save(params.rundir .. '/filters_enc_' .. t .. '.jpg', de)
      if params.display then
         _win1_ = image.display{image=dd, win=_win1_, legend='Decoder filters', zoom=2}
         _win2_ = image.display{image=de, win=_win2_, legend='Encoder filters', zoom=2}
      end

      -- store model
      torch.save(params.rundir .. '/model_' .. t .. '.bin', module)
      -- write training error
      torch.save(params.rundir .. '/error.mat', avTrainingError[{ {1,t/params.statinterval} }])
      -- write # of iterations
      torch.save(params.rundir .. '/iter.mat', avFistaIterations[{ {1,t/params.statinterval} }])

      -- update learning rate with decay
      currentLearningRate = params.eta/(1+(t/params.statinterval)*params.decay)
      err = 0
      iter = 0
   end
end
