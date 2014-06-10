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
print(sys.COLORS.red ..  '==> define parameters')

-- 2-class problem: faces!
local noutputs = 2

-- input dimensions: faces!
local nfeats = 1
local width = 32
local height = 32

-- hidden units, filter sizes (for ConvNet only):
local nstates = {16,32}
local filtsize = {5, 7}
local poolsize = 4

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')

local CNN = nn.Sequential()

-- stage 1: conv+max
CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2: conv+max
CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2]))
CNN:add(nn.Threshold())

local classifier = nn.Sequential()
-- stage 3: linear
classifier:add(nn.Reshape(nstates[2]))
classifier:add(nn.Linear(nstates[2], 2))

-- stage 4 : log probabilities
classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end

local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

