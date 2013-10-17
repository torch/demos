----------------------------------------------------------------------
-- Create CNN and loss to optimize.
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
print '==> construct CNN'

local CNN = nn.Sequential()

-- stage 1: conv+max
CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2: conv+max
CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

local classifier = nn.Sequential()
-- stage 3: linear
classifier:add(nn.Reshape(nstates[2]*filtsize*filtsize))
classifier:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
classifier:add(nn.Threshold())

-- stage 4: linear (classifier)
--CNN:add(dropout)
classifier:add(nn.Linear(nstates[3], noutputs))

-- stage 5 : log probabilities
classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end
for _,layer in ipairs(classifier.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #classifier.modules-1 then
         layer.bias:zero()
      end
   end
end


model = nn.Sequential()
model:add(CNN)
model:add(classifier)
-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print '==> here is the CNN:'
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

