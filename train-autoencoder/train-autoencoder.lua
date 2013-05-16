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
-- Note: simple auto-encoders (with no sparsity constraint on the code) typically
-- don't yield filters that are visually appealing, although they might be
-- minimizing the reconstruction error correctly.
--
-- We demonstrate 2 types of auto-encoders:
--   * plain: regular auto-encoder
--   * predictive sparse decomposition (PSD): the encoder is trained
--     to predict an optimal sparse decomposition of the input
--
-- Both types of auto-encoders can use linear or convolutional
-- encoders/decoders. The convolutional version typically yields more
-- interesting, less redundant filters for images.
--
-- Koray Kavukcuoglu, Clement Farabet
----------------------------------------------------------------------

require 'unsup'
require 'image'
require 'optim'
require 'autoencoder-data'

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 2, 'threads')

-- for all models:
cmd:option('-model', 'conv-psd', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 16, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 1, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-maxiter', 1000000, 'max number of updates')

-- use hessian information for training:
cmd:option('-hessian', true, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 10000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 500, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 9, 'size of convolutional kernels')

-- logging:
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')
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
cmd:addTime('psd')
cmd:log(params.rundir .. '/log.txt', params)

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

   -- verbose
   print('==> constructed linear auto-encoder')

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

   -- verbose
   print('==> constructed convolutional auto-encoder')

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

   -- verbose
   print('==> constructed linear predictive sparse decomposition (PSD) auto-encoder')

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

   -- verbose
   print('==> constructed convolutional predictive sparse decomposition (PSD) auto-encoder')

else
   print('==> unknown model: ' .. params.model)
   os.exit()
end

----------------------------------------------------------------------
-- trainable parameters
--

-- are we using the hessian?
if params.hessian then
   module:initDiagHessianParameters()
end

-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()

----------------------------------------------------------------------
-- train model
--

print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0

for t = 1,params.maxiter,params.batchsize do

   --------------------------------------------------------------------
   -- update diagonal hessian parameters
   --
   if params.hessian and math.fmod(t , params.hessianinterval) == 1 then
      -- some extra vars:
      local hessiansamples = params.hessiansamples
      local minhessian = params.minhessian
      local maxhessian = params.maxhessian
      local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
      etas = etas or ddl_ddx:clone()

      print('==> estimating diagonal hessian elements')
      for i = 1,hessiansamples do
         -- next
         local ex = dataset[i]
         local input = ex[1]
         local target = ex[2]
         module:updateOutput(input, target)

         -- gradient
         dl_dx:zero()
         module:updateGradInput(input, target)
         module:accGradParameters(input, target)

         -- hessian
         ddl_ddx:zero()
         module:updateDiagHessianInput(input, target)
         module:accDiagHessianParameters(input, target)

         -- accumulate
         ddl_ddx_avg:add(1/hessiansamples, ddl_ddx)
      end

      -- cap hessian params
      print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
      ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
      ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
      print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())

      -- generate learning rates
      etas:fill(1):cdiv(ddl_ddx_avg)
   end

   --------------------------------------------------------------------
   -- progress
   --
   iter = iter+1
   xlua.progress(iter, params.statinterval)

   --------------------------------------------------------------------
   -- create mini-batch
   --
   local example = dataset[t]
   local inputs = {}
   local targets = {}
   for i = t,t+params.batchsize-1 do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      table.insert(inputs, input)
      table.insert(targets, target)
   end

   --------------------------------------------------------------------
   -- define eval closure
   --
   local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- estimate f and gradients, for minibatch
      for i = 1,#inputs do
         -- f
         f = f + module:updateOutput(inputs[i], targets[i])

         -- gradients
         module:updateGradInput(inputs[i], targets[i])
         module:accGradParameters(inputs[i], targets[i])
      end

      -- normalize
      dl_dx:div(#inputs)
      f = f/#inputs

      -- return f and df/dx
      return f,dl_dx
   end

   --------------------------------------------------------------------
   -- one SGD step
   --
   sgdconf = sgdconf or {learningRate = params.eta,
                         learningRateDecay = params.etadecay,
                         learningRates = etas,
                         momentum = params.momentum}
   _,fs = optim.sgd(feval, x, sgdconf)
   err = err + fs[1]

   -- normalize
   if params.model:find('psd') then
      module:normalize()
   end

   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   if math.fmod(t , params.statinterval) == 0 then

      -- report
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)

      -- get weights
      eweight = module.encoder.modules[1].weight
      if module.decoder.D then
         dweight = module.decoder.D.weight
      else
         dweight = module.decoder.modules[1].weight
      end

      -- reshape weights if linear matrix is used
      if params.model:find('linear') then
         dweight = dweight:transpose(1,2):unfold(2,params.inputsize,params.inputsize)
         eweight = eweight:unfold(2,params.inputsize,params.inputsize)
      end

      -- render filters
      dd = image.toDisplayTensor{input=dweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(params.nfiltersout)),
                                 symmetric=true}
      de = image.toDisplayTensor{input=eweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(params.nfiltersout)),
                                 symmetric=true}

      -- live display
      if params.display then
         _win1_ = image.display{image=dd, win=_win1_, legend='Decoder filters', zoom=2}
         _win2_ = image.display{image=de, win=_win2_, legend='Encoder filters', zoom=2}
      end

      -- save stuff
      image.save(params.rundir .. '/filters_dec_' .. t .. '.jpg', dd)
      image.save(params.rundir .. '/filters_enc_' .. t .. '.jpg', de)
      torch.save(params.rundir .. '/model_' .. t .. '.bin', module)

      -- reset counters
      err = 0; iter = 0
   end
end
