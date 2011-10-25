#!/usr/bin/env qlua

require 'XLearn'
require 'os'

-- need QT
toolBox.useQT()

-- resource file
local ui = paths.thisfile('face-detect.ui')

-- Camera frames
local source = nn.ImageSource('camera')
local widget = qtuiloader.load(ui)
local painter = qt.QtLuaPainter(widget.frame)

local vslide
local hslide

local time = os.clock()
local timeNetwork
local fps
local diff

-- retrieve trained network
local convnet = nn.Sequential()
local file = torch.DiskFile('../trained-nets/network-face-detect', 'r')
convnet:read(file)
file:close()

-- add tensors for scales
local rgb2y = nn.ImageTransform('rgb2y')
local frameY
local scales = {1/3, 1/5, 1/7, 1/10}
local outputMaps
local listOfFaces

-- to dump images
local imagesDumped = 0

------------------------------
-- routines
------------------------------

-- demo proper
local function process()
   camFrame = source:forward()

   -- time
   diff = os.clock() - time
   time = os.clock()
   fps = 1/diff
   
   -- get frame from camera, convert to YUV, and just keep Y
   frameY = rgb2y:forward(camFrame)

   -- pyramid of scales
   local pyramid = image.makePyramid{tensor=frameY, scales=scales}

   -- forward prop
   listOfFaces = nil
   outputMaps = {}
   i = 1
   timeNetwork = os.clock()
   while true do
      local img = table.remove(pyramid, 1)
      if img == nil then break end
      -- forward prop through convnet
      local result = convnet:forward(img)
      -- process output map
      listOfFaces = image.findBlobs{tensor=result, threshold=0.02*vslide-1, 
                                    maxsize=10, discardClass=2, scale=scales[i],
                                    labels={"face"},
                                    listOfBlobs=listOfFaces}
      -- store result
      local resulttemp = torch.Tensor( result:size() ):copy(result)
      table.insert(outputMaps, resulttemp)
      i=i+1
   end
   timeNetwork = os.clock() - timeNetwork

   -- Extract Centroids of detections
   listOfFaces = image.reorderBlobs(listOfFaces)
   listOfFaces = image.remapBlobs(listOfFaces)
   listOfFaces = image.mergeBlobs(listOfFaces, 50)
end

local function getPatch(args)
   local image = args.image
   local x = args.x 
   local y = args.y 
   local w = args.w
   local h = args.h

   -- check boundaries
   w = math.min(image:size(1) - x, w)
   h = math.min(image:size(2) - y, h)
   return image:narrow(1,x,w):narrow(2,y,h)
end

local function paint()
   local zoom = 1/hslide
   painter:gbegin()
   painter:showpage()
   painter:setcolor("black")
   painter:setfont(qt.QFont{serif=false,italic=false,size=12})

   -- image from cam
   image.qtdisplay{ tensor=frameY, painter=painter, 
                    globalzoom=zoom, 
                    min=0, max=1,
                    inplace=true,
                    offset_x=0, offset_y=0,
                    legend='Face detection'}

   -- disp FPS
   painter:moveto(10, frameY:size(2)*2*zoom-30)
   painter:show(string.format('Time [all] = %0f', diff))
   painter:moveto(10, frameY:size(2)*2*zoom)
   painter:show(string.format('Time [net] = %0f', timeNetwork))

   if widget.checkBox1.checked then
      -- image pre-processed
      image.qtdisplay{ tensor=convnet.modules[1].output, painter=painter, 
                       globalzoom=zoom, 
                       zoom=4,
                       min=-1, max=1,
                       inplace=true,
                       offset_x=0, offset_y=frameY:size(2),
                       legend='Local Normalized'}

      local step = 40
      for i=1,#outputMaps do
         local map = table.remove(outputMaps, 1)
         image.qtdisplay{ tensor=map:narrow(3,1,1), painter=painter, 
                          globalzoom=zoom,
                          zoom = 8,
                          inplace=true,
                          min=-1, max=1,
                          offset_x=800, 
                          offset_y=step}
         step = map:size(2)*8 + step + 40
      end
   end

   -- print objects
   local i = 1
   local done = 0
   while true do
      if (listOfFaces[i] ~= nil) then
         local x = listOfFaces[i].x
         local y = listOfFaces[i].y
         local scale = listOfFaces[i].scale
         image.qtdrawbox{ painter=painter,
                          x=x * 4,
                          y=y * 4,
                          w=32/scale,
                          h=32/scale,
                          globalzoom=zoom, 
                          legend=listOfFaces[i].tag}
         done = done + 1
         
         -- option: dump false negatives to file (assumes that no face is in the fov)
         if widget.checkBox4.checked then
            local falsePositive = getPatch{image=frameRGB, 
                                           x=x/scale*4, y=y/scale*4, 
                                           w=32/scale, h=32/scale}
            -- resize patch
            local resizePatch = torch.Tensor( 32, 32, 3 )
            image.scale(falsePositive, resizePatch, 'simple')
            
            -- rescale patch
            local scaledPatch = image.yuv2file(resizePatch)
            
            -- save patch
            image.savePNG('scratch/false-positive-'..os.date("%Y-%m-%d-%X")..
                       '-'..imagesDumped..'.png', scaledPatch)
            imagesDumped = imagesDumped + 1
         end

         -- dump network internals
         if widget.checkBox2.checked then
            pyramid = image.makePyramid{tensor=frameY, scales={1/2}}
            img = table.remove(pyramid, 1)
            image.displayNetwork{network=convnet,
                                 maps=true,
                                 filters=true,
                                 input=img,
                                 dumpfile='scratch/face-net-'..os.date("%Y-%m-%d-%X")..'.png'}
         end
      end
      i = i + 1
      if (done == listOfFaces.nbOfBlobs) then
         break
      end
   end
   painter:gend()
end


function demo()
   -- Loop Process
   local timer = qt.QTimer()
   timer.interval = 0
   timer.singleShot = false
   timer:start()
   qt.connect(timer,'timeout()',
              function() 
                 vslide = widget.verticalSlider.value
                 hslide = widget.horizontalSlider.value
                 process()
                 if widget.checkBox3.checked then
                    paint()
                 end
                 timer:start()
              end )
   
   -- Close Process   
   widget.windowTitle = "Live Processing"
   widget:show()

end

demo()

