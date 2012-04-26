#!/usr/bin/env qlua
------------------------------------------------------------
-- a face detector, based on a simple convolutional network,
-- trained end-to-end for that task.
--
-- Clement Farabet
--

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
xrequire('inline',true)
xrequire('camera',true)
xrequire('nnx',true)

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, you can specify the camera index: /dev/videoIDX', 
          default=0}
op:option{'-n', '--network', action='store', dest='network', 
          help='path to existing [trained] network',
          default='face.net'}
opt,args = op:parse()

-- blob parser
parse = inline.load [[
      // get args
      const void* id = luaT_checktypename2id(L, "torch.DoubleTensor");
      THDoubleTensor *tensor = luaT_checkudata(L, 1, id);
      double threshold = lua_tonumber(L, 2);
      int table_blobs = 3;
      int idx = lua_objlen(L, 3) + 1;
      double scale = lua_tonumber(L, 4);

      // loop over pixels
      int x,y;
      for (y=0; y<tensor->size[0]; y++) {
         for (x=0; x<tensor->size[1]; x++) {
            double val = THDoubleTensor_get2d(tensor, y, x);
            if (val > threshold) {
               // entry = {}
               lua_newtable(L);
               int entry = lua_gettop(L);

               // entry[1] = x
               lua_pushnumber(L, x);
               lua_rawseti(L, entry, 1);

               // entry[2] = y
               lua_pushnumber(L, y);
               lua_rawseti(L, entry, 2);

               // entry[3] = scale
               lua_pushnumber(L, scale);
               lua_rawseti(L, entry, 3);

               // blobs[idx] = entry; idx = idx + 1
               lua_rawseti(L, table_blobs, idx++);
            }
         }
      }
      return 0;
]]

-- load pre-trained network from disk
network = nn.Sequential()
network = torch.load(opt.network)
network_fov = 32
network_sub = 4

-- setup camera
camera = image.Camera(opt.camidx)

-- process input at multiple scales
scales = {0.3, 0.24, 0.192, 0.15, 0.12, 0.1}

-- use a pyramid packer/unpacker
require 'PyramidPacker'
require 'PyramidUnPacker'
packer = nn.PyramidPacker(network, scales)
unpacker = nn.PyramidUnPacker(network)

-- setup GUI (external UI file)
if not win or not widget then 
   widget = qtuiloader.load('g.ui')
   win = qt.QtLuaPainter(widget.frame) 
end

-- a gaussian for smoothing the distributions
gaussian = image.gaussian(3,0.15)

-- profiler
p = xlua.Profiler()

-- process function
function process()
   -- (1) grab frame
   frame = camera:forward()

   -- (2) transform it into Y space
   frameY = image.rgb2y(frame)

   -- (3) create multiscale pyramid
   pyramid, coordinates = packer:forward(frameY)

   -- (4) run pre-trained network on it
   multiscale = network:forward(pyramid)

   -- (5) unpack pyramid
   distributions = unpacker:forward(multiscale, coordinates)

   -- set up and display the threshold
   widget.verticalSlider.maximum = 200
   threshold = widget.verticalSlider.value/100
   widget.lcdNumber.value = threshold

   -- (6) parse distributions to extract blob centroids
   max_global = 0
   rawresults = {}
   for i,distribution in ipairs(distributions) do
      local smoothed = image.convolve(distribution[1]:add(1):mul(0.5), gaussian)
      parse(smoothed, threshold, rawresults, scales[i])

      -- compute Max. value of the output layer
      local max_col = torch.max(smoothed,1)
      local max_val, idx_col = torch.max(max_col, 2)
      if max_global < max_val[1][1] then
         max_global = max_val[1][1]
      end
      widget.progressBar.value = math.floor(max_global*10000/widget.verticalSlider.maximum+0.5)
   end

   -- (7) clean up results
   k=0
   duplicate=0
   detections = {}
   for i,res in ipairs(rawresults) do
      local scale = res[3]
      local x = res[1]*network_sub/scale
      local y = res[2]*network_sub/scale
      local w = network_fov/scale
      local h = network_fov/scale
      
      -- kill excessive detection rectangles
      for m=1, k do
         if (detections[m].x<=x) and x<=(detections[m].x+detections[m].w) and (detections[m].y<=y) and (y<=(detections[m].y+detections[m].h)) then 
            duplicate=1								  			
         end
         if (detections[m].x>=x)and (x+w)>=detections[m].x and (detections[m].y>=y) and (y + h)>=detections[m].y  then 
            duplicate=1
         end
      end
      if (duplicate==0) then	
         k=k+1	
         detections[k] = {x=x, y=y, w=w, h=h}
   	  end
      duplicate=0
   end
end

-- display function
function display()
   win:gbegin()
   win:showpage()
   -- (1) display input image + pyramid
   image.display{image=frame, win=win}

   -- (2) overlay bounding boxes for each detection
   for i,detect in ipairs(detections) do
      win:setcolor(1,0,0)
      win:rectangle(detect.x, detect.y, detect.w, detect.h)
      win:stroke()
      win:setfont(qt.QFont{serif=false,italic=false,size=16})
      win:moveto(detect.x, detect.y-1)
      win:show('face')
   end

   -- (3) display distributions
   local prevx = 0
   for i,distribution in ipairs(distributions) do
      local prev = distributions[i-1]
      if prev then prevx = prevx + prev:size(3) end
      image.display{image=distribution[1], win=win, x=prevx, min=0, max=1}
   end

   win:gend()
end

-- setup gui
timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              p:start('prediction')
              process()
              p:lap('prediction')
              p:start('display')
              display()
              p:lap('display')
              require 'openmp'
              timer:start()
              p:printAll()
           end)
widget.windowTitle = 'Face Detector'
widget:show()
timer:start()
