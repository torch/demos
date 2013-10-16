----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem: faces!
local noutputs = 2

-- input dimensions: faces!
local nfeats = 1
local width = 32
local height = 32

-- hidden units, filter sizes (for ConvNet only):
local nstates = {8,16,32}
local filtsize = 5
local poolsize = 2

-- dropout?
--local dropout = nn.Dropout(opt.dropout)


----------------------------------------------------------------------
print '==> construct model'

local model = nn.Sequential()

-- stage 1: conv+max
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Threshold())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2: conv+max
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Threshold())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3: linear
model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Threshold())

-- stage 4: linear (classifier)
--model:add(dropout)
model:add(nn.Linear(nstates[3], noutputs))

-- stage 5 : log probabilities
model:add(nn.LogSoftMax())

-- Loss: NLL
loss = nn.ClassNLLCriterion()

-- adjust all biases for threshold activation units
for _,layer in ipairs(model.modules) do
   if layer.bias then
      layer.bias:add(.1)
   end
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
   --dropout = dropout
}

