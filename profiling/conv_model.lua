----------------------------------------------------------------------
-- E. Culurciello 
-- test for Julian Ibarz <julianibarz@google.com>
-- January 10th 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> define parameters'

-- # of classes at output
noutputs = 2

-- input dimensions
nfeats = 3 -- 3 colors/planes input image
width = 50 -- input window size
height = 50
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = {16,64, 64*16, 16*16, noutputs} -- filters number / neurons in each layer
filtsize = 9
poolsize = 4
--node = {64,16}
--depth = {16, 16}
normkernel = image.gaussian1D(3)


-- desired image size:
im_pl = 3
im_sz = 2200
testima = torch.Tensor(im_pl, im_sz, im_sz) -- test image of 5 Mpix 3 colors/planes

----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

---- stage 3 : filter bank -> squashing -> L2 pooling -> normalization
--model:add(nn.SpatialConvolution(nn.tables.random(nstates[2], nstates[3], fanin[3]), filtsize, filtsize))
--model:add(nn.Tanh())
--model:add(nn.SpatialLPPooling(nstates[3],2,poolsize,poolsize,poolsize,poolsize))
--model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 4 : fully connected layer
in_size =  ((im_sz-9+1)/4 - 9 +1)/4 --nstates[2]*filtsize*filtsize
--out_size = node[1] * depth[1]

model:add(nn.Reshape(nstates[2]*in_size^2))
model:add(nn.Linear(nstates[2]*in_size^2, nstates[3]))
model:add(nn.Tanh())

-- stage 5 : fully connected layer
model:add(nn.Linear(nstates[3], nstates[4]))
model:add(nn.Tanh())

-- linear output.
model:add(nn.Linear(nstates[4], noutputs))
-- logreg?
model:add(nn.Sigmoid())

print '==> testing model on image'
time = sys.clock()
outima = model:forward(testima)
time = sys.clock() - time
print('==> Compute Time = ' .. (time*1000) .. 'ms')
