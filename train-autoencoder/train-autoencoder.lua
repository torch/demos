----------------------------------------------------------------------
-- This script shows how to train autoencoders on natural images,
-- using the unsup package.
--
-- Borrowed from Koray Kavukcuoglu's unsup demos
--
-- In this script, we demonstrate the use of different types of
-- autoencoders. Learned filters can be visualized by providing the
-- flag -display.
--
-- A few configurations that seem to work:

-- Linear auto-encoder:
-- $ torch train-autoencoder.lua -display -model linear -eta 0.1 -lambda 1

-- Linear PSD auto-encoder:
-- $ torch train-autoencoder.lua -display -model linear-psd -eta 0.1 -lambda 0.5

-- Convolutional PSD auto-encoder:
-- $ torch train-autoencoder.lua -display -model conv-psd -eta 0.001 -lambda 0.02

-- Note: simple auto-encoders (with no sparsity constraint on the code) typically
-- don't yield filters that are visually appealing, although they might be
-- minimizing the reconstruction error correctly.

-- Clement Farabet
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
-- general options:
cmd:option('-dir','outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 123211, 'initial random seed')
cmd:option('-threads', 2, 'threads')

-- for all models:
cmd:option('-model', 'linear', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsize', 12, 'size of each input patch')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 256, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 1e-1, 'learning rate')
cmd:option('-eta_encoder', 0, 'encoder learning rate')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-decay', 0, 'weight decay')
cmd:option('-maxiter', 1000000, 'max number of updates')

-- for conv models:
cmd:option('-kernelsize', 12, 'size of convolutional kernels')

-- logging:
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-statinterval', 1000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', false, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
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
   displayData(dataset, 100, 10, 2)
end

----------------------------------------------------------------------
-- create model
--
if params.model == 'linear' then

   -- params
   inputSize = params.inputsize*params.inputsize
   outputSize = params.nfiltersout

   -- encoder
   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputSize,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder
   decoder = nn.Sequential()
   decoder:add(nn.Linear(outputSize,inputSize))

   -- complete model
   module = unsup.AutoEncoder(encoder, decoder, params.beta)

elseif params.model == 'conv' then

   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputsize, params.inputsize

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputFeatures))

   -- decoder:
   decoder = nn.Sequential()
   decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))

   -- complete model
   module = unsup.AutoEncoder(encoder, decoder, params.beta)

   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()

elseif params.model == 'linear-psd' then

   -- params
   inputSize = params.inputsize*params.inputsize
   outputSize = params.nfiltersout

   -- encoder
   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputSize,outputSize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputSize))

   -- decoder is L1 solution
   decoder = unsup.LinearFistaL1(inputSize, outputSize, params.lambda)

   -- PSD autoencoder
   module = unsup.PSD(encoder, decoder, params.beta)

elseif params.model == 'conv-psd' then

   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputsize, params.inputsize

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputFeatures))

   -- decoder is L1 solution:
   decoder = unsup.SpatialConvFistaL1(decodertable, kw, kh, iw, ih, params.lambda)

   -- PSD autoencoder
   module = unsup.PSD(encoder, decoder, params.beta)

   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()

else
   print('unknown model: ' .. params.model)
   os.exit()
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
      local err = module:updateOutput(input, target)
      module:zeroGradParameters()
      module:updateGradInput(input, target)
      module:accGradParameters(input, target)
      module:updateParameters(currentLearningRate)
      return err
   end
   local serr, siter = updateSample()
   err = err + serr

   -- gather/print statistics
   if math.fmod(t , params.statinterval) == 0 then
      -- training error / iteration
      avTrainingError[t/params.statinterval] = err/params.statinterval

      -- report
      print('# iter=' .. t .. ' eta = ( ' .. currentLearningRate[1] .. 
            ', ' .. currentLearningRate[2] .. ' ) current error = ' .. err)

      -- plot filters
      if params.model == 'linear' then
         dd = image.toDisplayTensor{input=module.decoder.modules[1].weight:transpose(1,2):unfold(2,params.inputsize,params.inputsize),padding=2,nrow=16,symmetric=true}
         de = image.toDisplayTensor{input=module.encoder.modules[1].weight:unfold(2,params.inputsize,params.inputsize),padding=2,nrow=16,symmetric=true}
      elseif params.model == 'conv' then
         de = image.toDisplayTensor{input=module.encoder.modules[1].weight,padding=2,nrow=16,symmetric=true}
         dd = image.toDisplayTensor{input=module.decoder.modules[1].weight,padding=2,nrow=16,symmetric=true}
      elseif params.model == 'conv-psd' then
         de = image.toDisplayTensor{input=module.encoder.modules[1].weight,padding=2,nrow=16,symmetric=true}
         dd = image.toDisplayTensor{input=module.decoder.D.weight,padding=2,nrow=16,symmetric=true}
      elseif params.model == 'linear-psd' then
         dd = image.toDisplayTensor{input=module.decoder.D.weight:transpose(1,2):unfold(2,params.inputsize,params.inputsize),padding=2,nrow=16,symmetric=true}
         de = image.toDisplayTensor{input=module.encoder.modules[1].weight:unfold(2,params.inputsize,params.inputsize),padding=2,nrow=16,symmetric=true}
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

      -- update learning rate with decay
      currentLearningRate = params.eta/(1+(t/params.statinterval)*params.decay)
      err = 0
      iter = 0
   end
end
