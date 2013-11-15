#!/usr/bin/env torch
------------------------------------------------------------
-- 
-- CNN face detector, based on convolutional network nets
--
-- original: Clement Farabet
-- E. Culurciello, A. Dundar
-- Mon Oct 14 14:58:50 EDT 2013
--
------------------------------------------------------------

require 'pl'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
--require 'camera'
require 'nnx'
require 'image'

print '==> processing options'

opt = lapp[[
   -c, --camidx   (default 0)       camera index: /dev/videoIDX
   -n, --network  (default 'face.net')    path to network
   -x, --runnnx   (default true)    run on hardware nn_X 
   -t, --threads  (default 8)       number of threads
]]

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

-- blob parser:
function parse(tin, threshold, blobs, scale)
  --loop over pixels
  for y=1, tin:size(1) do
     for x=1, tin:size(2) do
        local val = tin[y][x]
        if (val > threshold) then               
          entry = {}
          entry[1] = x
          entry[2] = y
          entry[3] = scale
          table.insert(blobs,entry)
      end
    end
  end
end

-- load pre-trained network from disk
network = torch.load(opt.network):float()

classifier1 = nn.Sequential()
classifier1:add(network.modules[6]:clone())
classifier1:add(network.modules[7]:clone())
classifier = nn.SpatialClassifier(classifier1)
network.modules[6] = nn.SpatialClassifier(classifier1)
network.modules[7]=nil
network_fov = 32
network_sub = 4

-- remove SpatialCconvolutionMM:
m1 = network.modules[1]:clone()
network.modules[1] = nn.SpatialConvolution(1,8,5,5)
network.modules[1].weight = m1.weight:reshape(8,1,5,5)
network.modules[1].bias = m1.bias
m1 = network.modules[4]:clone()
network.modules[4] = nn.SpatialConvolution(8,64,7,7)
network.modules[4].weight = m1.weight:reshape(64,8,7,7)
network.modules[4].bias = m1.bias

-- setup camera
--camera = image.Camera(opt.camidx)

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

-- profiler
p = xlua.Profiler()

-- process function
function process()
   -- (1) grab frame
   frame = image.lena()--camera:forward()

   -- (2) transform it into Y space
   frameY = frame[2]:reshape(1,512,512) -- just green component
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
   threshold = 0.9--widget.verticalSlider.value/100

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
      print(detect.x, detect.y, detect.w, detect.h)
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
