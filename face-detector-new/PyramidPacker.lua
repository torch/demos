local PyramidPacker, parent = torch.class('nn.PyramidPacker', 'nn.Module')

function getCoordinates(args)
   local scales = args.scales
   local step_width = args.step_width
   local step_height = args.step_height
   local dim_width_orig = args.dim_width
   local dim_height_orig = args.dim_height

   local dim_width = math.floor(dim_width_orig*scales[1])
   local dim_height = math.floor(dim_height_orig*scales[1])
   -- we define the coordinates table, which we will fill-in
   -- once per each different input or different scales
   -- and we will use it to pack and unpack different sclales into/out of
   -- one big pack.
   -- The rows of the table are different scales,
   -- the columns of the table are:
   --     1   2   3   4   5     6
   --     x1  y1  x2  y2  width height
   --
   -- (x1, y1) - top left corner, (x2, y2) - bottom right corner,
   -- (width, height) - sizes of the current scale

   local coordinates = torch.Tensor(#scales, 6)
   coordinates[1][1] = 1
   coordinates[1][2] = 1
   coordinates[1][3] = dim_width
   coordinates[1][4] = dim_height
   coordinates[1][5] = dim_width
   coordinates[1][6] = dim_height
   local max_width = dim_width
   local max_height = dim_height

   -- fill the coordinates table and get the size for the big pack
   for i=2,#scales,1 do

      dim_width = math.floor(dim_width_orig*scales[i])
      dim_height = math.floor(dim_height_orig*scales[i])

      -- an even case - putting down
      if (i%2 == 0) then
         coordinates[i][1] = coordinates[i-1][1]
         coordinates[i][2] = (math.floor((coordinates[i-1][4]-1)/step_height) + 1)*step_height+1
      else -- an odd case - putting beside
         coordinates[i][1] = (math.floor((coordinates[i-1][3]-1)/step_width) + 1)*step_width+1
         coordinates[i][2] = coordinates[i-1][2]
      end

      coordinates[i][3] = dim_width + coordinates[i][1] - 1
      coordinates[i][4] = dim_height + coordinates[i][2] - 1
      coordinates[i][5] = dim_width
      coordinates[i][6] = dim_height

      max_width = math.max(max_width, coordinates[i][3])
      max_height = math.max(max_height, coordinates[i][4])
   end

   return coordinates, max_width, max_height
end

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

function PyramidPacker:__init(network, scales)
   parent.__init(self)
   -- vars
   self.scales = scales or {1}
   self.dim_width = 1
   self.dim_height = 1
   self.dimz = 1
   if network then
      -- infer params from given net
      self.step_width, self.step_height = getGlobalSizes({sizes_tbl=getSizesTbl(network)})--.modules[1])})
   else
      self.step_width = 1
      self.step_height = 1
   end

   self.output = torch.Tensor(1,1,1)
   self.output:fill(0)
end

function PyramidPacker:forward(input)
   self.step_height = 4
   self.step_width = 4
   if ((input:size(3) ~= self.dim_width) or (input:size(2) ~= self.dim_height)) then
      self.dim_height = input:size(2)
      self.dim_width = input:size(3)
      self.coordinates, self.max_width, self.max_height =
         getCoordinates({dim_width = self.dim_width, dim_height = self.dim_height,
                         scales = self.scales,
                         step_width = self.step_width, step_height = self.step_height})
   end
   if(input:size(1) ~= dim_z) then self.dimz = input:size(1) end
   self.output:resize(self.dimz, self.max_height, self.max_width):zero()
   -- using the coordinates table fill the pack with different scales
   -- if the pack and coordinates already exist for the same input size we go directly to here
   for i = 1,#self.scales do
      local temp = self.output:narrow(3,self.coordinates[i][1],self.coordinates[i][5])
      temp = temp:narrow(2,self.coordinates[i][2],self.coordinates[i][6])
      image.scale(temp, input, 'bilinear')
   end

   return self.output, self.coordinates
end

function PyramidPacker:backward(input, gradOutput)
   xlua.error('backward non implemented', 'PyramidPacker')
end

function PyramidPacker:write(file)
   parent.write(self,file)
   file:writeDouble(#self.scales)
   for i = 1,#self.scales do
      file:writeDouble(self.scales[i])
   end
end

function PyramidPacker:read(file)
   parent.read(self,file)
   local nbScales = file:readDouble()
   for i = 1,nbScales do
      self.scales[i] = file:readDouble()
   end
end
