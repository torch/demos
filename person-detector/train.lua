----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to SGD.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss

----------------------------------------------------------------------
-- Save light network tools:
function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   module.fgradInput = nil
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining some tools')

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local x = torch.Tensor(opt.batchSize,trainData.data:size(2), 
         trainData.data:size(3), trainData.data:size(4)) --faces data
local yt = torch.Tensor(opt.batchSize)
if opt.type == 'cuda' then 
   x = x:cuda()
   yt = yt:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch

local function train(trainData)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[shuffle[i]]
         yt[idx] = trainData.labels[shuffle[i]]
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(x)
         local E = loss:forward(y,yt)

         -- estimate df/dW
         local dE_dy = loss:backward(y,yt)   
         model:backward(x,dE_dy)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(y[i],yt[i])
         end

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.sgd(eval_E, w, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   model1 = model:clone()
   netLighter(model1)
   torch.save(filename, model1)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

-- Export:
return train
