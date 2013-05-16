
require 'sys'
require 'cunn'

cutorch.setDevice(arg[1] or 1)
print('DEVID = ' .. cutorch.getDevice())

bs = 512
ninputs = 4096
nhiddens = 4096
noutputs = 1000

n = nn.Sequential()
n:add( nn.Linear(ninputs, nhiddens) )
n:add( nn.Threshold() )
n:add( nn.Linear(nhiddens, noutputs) )
n:add( nn.Threshold() )

i = torch.randn(bs, ninputs)

ops = (ninputs*nhiddens + nhiddens*noutputs) * 2 * bs * 3

n:float()
i = i:float()

N=5

-- sys.tic()
-- for t = 1,N do
--    n:forward(i)
--    n:backward(i, n.output)
-- end
-- t = sys.toc()/N
-- print('MKL - GFLOP/s:', ops/t/1e9)

n:cuda()
i = i:cuda()

-- pre-alloc states:
n:forward(i)
n:backward(i, n.output)
cutorch.synchronize()

sys.tic()
for t = 1,N do
   n:forward(i)
   n:backward(i, n.output)
end
cutorch.synchronize()
t = sys.toc()/N
print('Fprop+Bprop+Acc - GFLOP/s:', ops/t/1e9)

