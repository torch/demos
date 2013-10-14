----------------------------------------------------------------------
-- Train a ConvNet on SVHN.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'pl'

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 0.2)         learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.5)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -q,--quantization       (default 2)         quantization of computational precision 8b=256, etc
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print '==> load modules'

local data  = require 'data'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print '==> training!'

while true do
   train(data.trainData)
   test(data.testData)
end

