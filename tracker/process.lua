require 'nnx'
local ffi = require("ffi")

ffi.cdef(io.open('fastdist.h', 'r'):read('*a'))
--ffi.cdef('#include "fastdist.h"')
fastdist = ffi.load("libfastdist.so")

encoder = torch.load(options.encoder)
encoder:float()
local mind_enc = encoder.modules[1].gradInput:size(1)
local minh_enc = encoder.modules[1].gradInput:size(2)
local minw_enc = encoder.modules[1].gradInput:size(3)
local boxh = options.boxh
local boxw = options.boxw
local downs = options.downs

print(' ... input window of convnet is ' .. mind_enc .. 'x' .. minh_enc .. 'x'  .. minw_enc .. ' in size')
if downs*1 == -1 then
   options.downs = math.min(boxh/minh_enc,boxw/minw_enc)
   downs = options.downs
   source.setdowns(downs)
   print(' ... setting downsampling to maximum of ' .. downs)
else
   print(' ... image downsampling ratio = ' .. downs)
end
print(' ... calibrating encoder so as to produce a single vector for a training patch of width ' .. boxw/downs .. ' and height ' .. boxh/downs)
local res = encoder:forward(torch.Tensor(3,boxh/downs,boxw/downs))
addpooler = nn.SpatialLPPooling(res:size(1),2,res:size(3),res:size(2),1,1)
encoderm  = encoder:clone()
encoderm:add(addpooler)
print(' ... appending a ' .. res:size(3) .. 'x' .. res:size(2) .. ' L2-pooling')
encoder_dw = 1
for i,mod in ipairs(encoderm.modules) do
   if mod.dW then encoder_dw = encoder_dw * mod.dW end
end
print(' ... encoder global downsampling ratio = ' .. encoder_dw)
print('')

encoder_full  = encoderm:clone()
encoder_patch = encoderm:clone()
profiler      = xlua.Profiler()

function GetMax(a)
	x,xi = torch.max(a,1)
	y,yi = torch.max(x,2)

	x_out = yi[1][1]
	y_out = xi[1][x_out]
	return y,x_out,y_out 
end

function GetMin(a)
   x,xi = torch.min(a,1)
   y,yi = torch.min(x,2)

   x_out = yi[1][1]
   y_out = xi[1][x_out]
   return y,x_out,y_out 
end

function RBFsimilarity(probMapOut, vectorMapIn, memory)
   local numProto = #memory
   local Vector = torch.Tensor(numProto, vectorMapIn:size(1), 1, 1):zero()
   local Weight = torch.Tensor(numProto):zero()
   local Std = torch.Tensor(numProto):zero()
   for i,proto in pairs(memory) do
      Vector[i] = proto.vector
      Weight[i] = proto.weight
      Std[i] = proto.std
   end

   probMapOut:zero()

   local metric = 'exponential'
   if metric == 'RBF' then
      fastdist.RBF(torch.data(vectorMapIn), vectorMapIn:size(1), vectorMapIn:size(2),
                   vectorMapIn:size(3), torch.data(probMapOut), Weight:size(1),
                   torch.data(Vector), torch.data(Weight), torch.data(Std))
   elseif metric == 'SMR' then
      fastdist.SMR(torch.data(vectorMapIn), vectorMapIn:size(1), vectorMapIn:size(2),
                   vectorMapIn:size(3), torch.data(probMapOut), Weight:size(1),
                   torch.data(Vector), torch.data(Weight), 0.2)
   elseif metric == 'exponential' then
      fastdist.exponential(torch.data(vectorMapIn), vectorMapIn:size(1), vectorMapIn:size(2),
                   vectorMapIn:size(3), torch.data(probMapOut), Weight:size(1),
                   torch.data(Vector), torch.data(Weight), torch.data(Std))
   end
end

-- grab camera frames, and process them
local function process()

   ------------------------------------------------------------
   -- (1) get a frame, and resize
   ------------------------------------------------------------
   profiler:start('get-frame')
   source:getframe()
   state.results = {}
   profiler:lap('get-frame')

   ------------------------------------------------------------
   -- (2) encode a full scene
   ------------------------------------------------------------
   profiler:start('process')
   denseFeatures = encoder_full:forward(state.procFrame)
   profiler:lap('process')

   ------------------------------------------------------------
   -- (3) generate a confidence map
   ------------------------------------------------------------
   profiler:start('generate-confidence-map')
   state.distributions:resize(#state.classes+1, denseFeatures:size(2), denseFeatures:size(3)):zero()
   if state.memory[1] then
      -- C code RBF function:
      RBFsimilarity(state.distributions[1], denseFeatures, state.memory[1])

      -- Or pure Lua dist functions:
      -- state.distributions[1]:zero()--:resize(denseFeatures:size(2), denseFeatures:size(3)):zero()
      -- -- match to prototype:
      -- for j = 1, denseFeatures:size(2) do
      --    for k = 1, denseFeatures:size(3) do
      --       state.distributions[1][j][k] = torch.dist(denseFeatures[{{},{j},{k}}], state.memory[1][1].vector:reshape(32))
      --    end
      -- end
   
   end
   profiler:lap('generate-confidence-map')
   
   ------------------------------------------------------------
   -- (4) estimate a new position 
   ------------------------------------------------------------
   profiler:start('estimate-new-position')
   value, px, py = GetMax(state.distributions[1]) -- get min if you use plain distance!
   state.maxProb = value[1][1]
   if state.memory[1] then
      local off_x = math.floor((state.rawFrame:size(3) - state.distributions:size(3)*downs*encoder_dw)/2)
      local off_y = math.floor((state.rawFrame:size(2) - state.distributions:size(2)*downs*encoder_dw)/2)

      local lx = (px-1) * downs * encoder_dw + 1 + off_x - boxw/2
      local ty = (py-1) * downs * encoder_dw + 1 + off_y - boxh/2
      -- make sure box is in frame
      lx = math.min(math.max(1,lx),state.rawFrame:size(3)-boxw+1)
      ty = math.min(math.max(1,ty),state.rawFrame:size(2)-boxh+1)

      local nresult = {lx=lx, ty=ty, cx=lx+boxw/2, cy=ty+boxh/2, w=boxw,
                       h=boxh, class=state.classes[1], id=1, source=4}
      if state.maxProb >= state.threshold then 
         table.insert(state.results, nresult)
      end
   end
   profiler:lap('estimate-new-position')

   ------------------------------------------------------------
   -- (6) capture new prototype, upon user request
   ------------------------------------------------------------
   if state.learn then
      profiler:start('learn-new-view')
      -- compute x,y coordinates
      local lx = math.min(math.max(state.learn.x-boxw/2,0),state.rawFrame:size(3)-boxw)
      local ty = math.min(math.max(state.learn.y-boxh/2,0),state.rawFrame:size(2)-boxh)
      state.logit('adding [' .. state.learn.class .. '] at ' .. lx 
                  .. ',' .. ty, state.learn.id)

      -- and create a result !!
      local nresult = {lx=lx, ty=ty, cx=lx+boxw/2, cy=ty+boxh/2, w=boxw, 
                       h=boxh, class=state.classes[state.learn.id], 
                       id=state.learn.id, source=6}
      table.insert(state.results, nresult)

      -- remap to smaller proc map
      lx = lx / downs + 1
      ty = ty / downs + 1

      -- store patch and its code
      patch = state.procFrame:narrow(3,lx,boxw/downs):narrow(2,ty,boxh/downs):clone()
      local code = encoder_patch:forward(patch):clone()

      state.memory[state.learn.id] = state.memory[state.learn.id] or {}
      table.insert(state.memory[state.learn.id], {patch=patch, vector=code, weight=1, std=options.std})

      -- done
      state.learn = nil
      profiler:lap('learn-new-view')
   end

   ------------------------------------------------------------
   -- (7) save results
   ------------------------------------------------------------
   if state.dsoutfile then
      local res = state.results[1]
      if res then
         state.dsoutfile:writeString(res.lx .. ',' .. res.ty .. ',' ..
                                     res.lx+res.w .. ',' .. res.ty+res.h)
      else
         state.dsoutfile:writeString('NaN,NaN,NaN,NaN')
      end
      state.dsoutfile:writeString('\n')
   end
end

return process
