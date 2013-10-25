#!/usr/bin/env torch
------------------------------------------------------------
-- 
-- CNN face detector, based on convolutional network nets
--
-- original: Clement Farabet
-- E. Culurciello
-- Mon Oct 14 14:58:50 EDT 2013
--
------------------------------------------------------------


require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'inline'
require 'camera'
require 'nnx'
require 'image'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='camera index: /dev/videoIDX', default=0}
op:option{'-n', '--network', action='store', dest='network', 
          help='path to existing [trained] network',
          default='model.net'}
opt,args = op:parse()

torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(8)

-- blob parser
parse = inline.load [[
      // get args
      const void* id = luaT_checktypename2id(L, "torch.FloatTensor");
      THFloatTensor *tensor = luaT_checkudata(L, 1, id);
      float threshold = lua_tonumber(L, 2);
      int table_blobs = 3;
      int idx = lua_objlen(L, 3) + 1;
      float scale = lua_tonumber(L, 4);

      // loop over pixels
      int x,y;
      for (y=0; y<tensor->size[0]; y++) {
         for (x=0; x<tensor->size[1]; x++) {
            float val = THFloatTensor_get2d(tensor, y, x);
            //printf("%f - %f ", val, threshold);
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
network = torch.load(opt.network):float()

-- replace classifier (2nd module) by SpatialClassifier
foveanet = network.modules[1]
classifier1 = network.modules[2]
classifier1.modules[3] = nil
classifier = nn.SpatialClassifier(classifier1)
network.modules[2] = classifier
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
   frame = camera:forward()   --image.lena()

   -- (2) transform it into Y space
   frameY = image.rgb2y(frame)
   mean = frameY:mean()
   std = frameY:std()
   frameY:add(-mean)
   frameY:div(std)

    -- (3) create multiscale pyramid
   pyramid, coordinates = packer:forward(frameY)
   -- (4) run pre-trained network on it
   multiscale = network:forward(pyramid)
   -- (5) unpack pyramid
   distributions = unpacker:forward(multiscale, coordinates)
   -- (6) parse distributions to extract blob centroids
   threshold = widget.verticalSlider.value/100

   rawresults = {}
   for i,distribution in ipairs(distributions) do
      parse(distribution[1], threshold, rawresults, scales[i])
   end

   -- (7) clean up results
   detections = {}
   for i,res in ipairs(rawresults) do
      local scale = res[3]
      local x = res[1]*network_sub/scale
      local y = res[2]*network_sub/scale
      local w = network_fov/scale
      local h = network_fov/scale
      detections[i] = {x=x, y=y, w=w, h=h}
   end
end

-- display function
function display()
   win:gbegin()
   win:showpage()
   -- (1) display input image + pyramid
   image.display{image=frame, win=win, saturation=false, min=0, max=1} 

   -- (2) overlay bounding boxes for each detection
   for i,detect in ipairs(detections) do
      win:setcolor(1,0,0)
      win:rectangle(detect.x, detect.y, detect.w, detect.h)
      win:stroke()
      win:setfont(qt.QFont{serif=false,italic=false,size=16})
      win:moveto(detect.x, detect.y-1)
      win:show('face')
   end
   win:gend()
end

-- setup gui
timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              p:start('full loop','fps')
              p:start('prediction','fps')
              process()
              p:lap('prediction')
              p:start('display','fps')
              display()
              p:lap('display')
              timer:start()
              p:lap('full loop')
              p:printAll()
           end)
widget.windowTitle = 'Face Detector'
widget:show()
timer:start()
