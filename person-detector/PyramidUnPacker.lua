
local PyramidUnPacker, parent = torch.class('nn.PyramidUnPacker', 'nn.Module')

local function getSizesTbl(net)
   local sizes_tbl = {}
   for i=1,#net.modules do
      dw = net.modules[i].dW
      dh = net.modules[i].dH
      kw = net.modules[i].kW
      kh = net.modules[i].kH
      if((dw ~= nil)and(dh ~= nil)and(kw ~= nil) and(kh ~= nil)) then 
	 table.insert(sizes_tbl, {kw=kw,kh=kh,dw=dw,dh=dh})
      end
   end

   return sizes_tbl
end

local function getRange(args)
   local sizes_tbl = args.sizes_tbl
   local idx_output = args.idx_output

   local x = torch.Tensor(#sizes_tbl+1)
   local y = torch.Tensor(#sizes_tbl+1)
   x[#sizes_tbl+1] = idx_output
   y[#sizes_tbl+1] = idx_output

   for k = #sizes_tbl,1,-1 do
      -- rightmost point of the image that affects x(k+1)
      x[k] = sizes_tbl[k].kw+ (x[k+1]-1) * sizes_tbl[k].dw
      -- leftmost point of the image that affects y(k+1)
      y[k] = 1 + (y[k+1]-1) * sizes_tbl[k].dw
   end
   local left_width = y[1]
   local right_width = x[1]

   for k = #sizes_tbl,1,-1 do
      -- rightmost point of the image that affects x(k+1)
      x[k] = sizes_tbl[k].kh+ (x[k+1]-1) * sizes_tbl[k].dh
      -- leftmost point of the image that affects y(k+1)
      y[k] = 1 + (y[k+1]-1) * sizes_tbl[k].dh
   end

   local left_height = y[1]
   local right_height = x[1]


   return left_width, right_width, left_height, right_height
end

local function getGlobalSizes(args)
   local sizes_tbl = args.sizes_tbl
   
   -- to find gobal kernel size we use recursive formula:
   -- glob_ker(n + 1) = 1
   -- glob_ker(n) = ker(n) + (glob_ker(n+1)-1)*step(n)
   --
   -- where: ker(n) - kernel size on layer n, step(n) - step size on layer n
   -- and n is number of layers that change the size of the input (convolution and subsample)
   local left_width1, right_width1, left_height1, right_height1 = getRange({sizes_tbl=sizes_tbl, idx_output=1})
   local ker_width = right_width1 - left_width1 +1
   local ker_height = right_height1 - left_height1 +1

   local step_width = 1
   local step_height = 1

   -- global step = MUL(step_1, step_2, ... , step_n)
   for i = 1, #sizes_tbl do
      step_width = step_width * sizes_tbl[i].dw
      step_height = step_height * sizes_tbl[i].dh
   end

   return step_width, step_height, ker_width, ker_height
end

function PyramidUnPacker:__init(network)
   parent.__init(self)

   -- infer params from given net
   self.step_width, self.step_height, self.ker_width, self.ker_height
      = getGlobalSizes({sizes_tbl=getSizesTbl(network)})
end

function PyramidUnPacker:forward(input, coordinates)
   self.out_tbl = {}
   self.coordinates = coordinates
   self.step_width = 4 
   self.step_height = 4
   self.ker_width = 46 -- size of CNN eye
   self.ker_height = 46
   for i = 1, self.coordinates:size(1) do
      local start_x = math.floor((self.coordinates[i][1] - 1)/self.step_width) + 1
      local start_y = math.floor((self.coordinates[i][2] - 1)/self.step_height) + 1
      local width = math.floor((self.coordinates[i][5] - self.ker_width)/self.step_width) + 1
      local height = math.floor((self.coordinates[i][6] - self.ker_height)/self.step_height) + 1
      local temp = input:narrow(3, start_x, width)
      temp = temp:narrow(2, start_y, height)
      table.insert(self.out_tbl, temp) 
   end
   return self.out_tbl
end

function PyramidUnPacker:backward(input, gradOutput)
   error('backward non implemented', 'PyramidUnPacker')
end

function PyramidUnPacker:write(file)
   parent.write(self,file)
   file:writeDouble(#self.scales)
   for i = 1,#self.scales do
      file:writeDouble(self.scales[i])
   end
end

function PyramidUnPacker:read(file)
   parent.read(self,file)
   local nbScales = file:readDouble()
   for i = 1,nbScales do
      self.scales[i] = file:readDouble()
   end
end
