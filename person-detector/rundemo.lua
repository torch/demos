#!/usr/bin/env torch
------------------------------------------------------------
--
-- People detector, based on convolutional network nets
--
-- E. Culurciello
-- Fri Jun 13 11:05:29 EDT 2014
--
------------------------------------------------------------

require 'pl'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'image'
require 'nnx'
require 'torchffi'

print '==> processing options'

opt = lapp[[
   -c, --camidx   (default 0)             camera index: /dev/videoIDX
   -n, --network  (default 'model.net')   path to networkimage.
   -t, --threads  (default 8)             number of threads
       --HD       (default true)          high resolution camera
]]

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

-- blob parser in FFI (almost as fast as pure C!):
-- does not work anymore after some Torch changes.... need fix!@
function parseFFI(pin, iH, iW, threshold, blobs, scale)
  --loop over pixels
  for y=0, iH-1 do
     for x=0, iW-1 do
        if (pin[iW*y+x] > threshold) then
          entry = {}
          entry[1] = x
          entry[2] = y
          entry[3] = scale
          table.insert(blobs,entry)
      end
    end
  end
end

function prune(detections)
  local pruned = {}

  local index = 1
   for i,detect in ipairs(detections) do
     local duplicate = 0
     for j, prune in ipairs(pruned) do
       -- if two detections left top corners are in close proximity discard one
       -- 50 is a proximity threshold can be changed 
       if (torch.abs(prune.x-detect.x)+torch.abs(prune.y-detect.y)<50) then
        duplicate = 1
       end
     end

     if duplicate == 0 then
      pruned[index] = {x=detect.x, y=detect.y, w=detect.w, h=detect.h}
      index = index+1
     end 
   end

   return pruned
end

-- load pre-trained network from disk
network1 = torch.load(opt.network) --load a network split in two: network and classifier
network1.modules[2].modules[5] = nil -- remove logsoftmax
network = network1.modules[1]:clone() -- split network
classifier1 = network1.modules[2]:clone() -- split and reconstruct classifier
classifier = nn.SpatialClassifier(classifier1)
network:add(classifier)
network_fov = 46
network_sub = 4

print('Neural Network used: \n', network) -- print final network

-- setup camera
local GUI
if opt.HD then
   camera = image.Camera(opt.camidx,640,360)
   GUI = 'HDg.ui'
else
   camera = image.Camera(opt.camidx)
   GUI = 'g.ui'
end

-- process input at multiple scales
scales = {0.5, 0.4, 0.3, 0.24, 0.192, 0.15}

-- use a pyramid packer/unpacker
require 'PyramidPacker'
require 'PyramidUnPacker'
packer = nn.PyramidPacker(network, scales)
unpacker = nn.PyramidUnPacker(network)

-- setup GUI (external UI file)
if not win or not widget then
   widget = qtuiloader.load(GUI)
   win = qt.QtLuaPainter(widget.frame)
end

-- profiler
p = xlua.Profiler()

-- process function
function process()
   -- (1) grab frame
   frame = camera:forward()

   -- (2) global normalization:
   local fmean = frame:mean()
   local fstd = frame:std()
   frame:add(-fmean)
   frame:div(fstd)
   
    -- (3) create multiscale pyramid
   pyramid, coordinates = packer:forward(frame)

   -- (4) run pre-trained network on it
   multiscale = network:forward(pyramid)
   -- (5) unpack pyramid
   distributions = unpacker:forward(multiscale, coordinates)
   -- (6) parse distributions to extract blob centroids
   threshold = widget.verticalSlider.value/100
   print(threshold)
  

   rawresults = {}
   -- function FFI:
   for i,distribution in ipairs(distributions) do
      local pdist = torch.data(distribution[1]:contiguous())
      parseFFI(pdist, distribution[1]:size(1), distribution[1]:size(2), threshold, rawresults, scales[i])
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

   detections = prune(detections)
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
      win:show('person')
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
              --p:printAll()
           end)
widget.windowTitle = 'e-Lab People Detector'
widget:show()
timer:start()
