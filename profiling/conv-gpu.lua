
-- libs:
require 'sys'
require 'cunn'

-- dev:
cutorch.setDevice(arg[1] or 1)
print('DEVID = ' .. cutorch.getDevice())

-- params:
batchSize = 128
isize = 64
ninputs = 32
nhiddens = 64
stride = 1
fsize = 8

-- batch input:
-- i = torch.randn(batchSize, ninputs, isize, isize)
i = torch.randn(ninputs, isize, isize, batchSize)
i = i:cuda()

-- layers to benchmark:
n = nn.Sequential()
-- n:add( nn.Transpose({1,4},{1,3},{1,2}) )
n:add( nn.SpatialConvolutionCUDA(ninputs, nhiddens, fsize, fsize, stride, stride) )
-- n:add( nn.Transpose({4,1},{4,2},{4,3}) )
n:cuda()

-- pre-alloc states:
n:forward(i)
n:backward(i, n.output)
cutorch.synchronize()

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
   n:backward(i,n.output)
end
cutorch.synchronize()
t = sys.toc()/nbOfAverages

-- result:
print('Fprop+Bprop+Acc - GFLOP/s:', ops/t/1e9)

