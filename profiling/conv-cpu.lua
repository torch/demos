
-- libs:
require 'sys'
require 'nnx'
--require 'cunn'

-- dev:
--utorch.setDevice(arg[1] or 1)
--print('DEVID = ' .. cutorch.getDevice())

-- params:
batchSize = 32
isize = 64
ninputs = 32
nhiddens = 64
stride = 1
fsize = 8

-- batch input:
i = torch.randn(batchSize, ninputs, isize, isize)

-- layers to benchmark:
n = nn.Sequential()
n:add( nn.SpatialConvolutionMM(ninputs, nhiddens, fsize, fsize, stride, stride) )

-- pre-alloc states:
n:forward(i)
n:backward(i, n.output)
--cutorch.synchronize()

-- nb of operations:
opsPerMAC = 2
steps = 3
ops = opsPerMAC * batchSize * steps * (
    ninputs*nhiddens*(fsize^2)*(((isize-fsize)/stride+1)^2) 
)

-- benchmark:
nbOfAverages = 3
sys.tic()
for t = 1,nbOfAverages do
   n:forward(i)
   n:backward(i, n.output)
end
--cutorch.synchronize()
t = sys.toc()/nbOfAverages

-- result:
print('Fprop+Bprop+Acc - GFLOP/s:', ops/t/1e9)

