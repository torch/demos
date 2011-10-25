#!/usr/bin/env qlua
----------------------------------------------------------------------
-- WHAT: executes a pretrained neural net on an image source.
--       the image source must be provided on the command line.
--       the detected object is then segmented, and the background
--       removed.
--
require 'XLearn'
require 'powerwatersegm'
require 'os'

-- parse args
op = OptionParser('%prog -s SOURCE [options]')
op:add_option{'-s', '--source', action='store', dest='source', 
              help='image source, can be one of: camera | lena'}
options,args = op:parse_args()

-- setup QT gui
toolBox.useQT()
widget = qtuiloader.load('face-detect.ui')
painter = qt.QtLuaPainter(widget.frame)

-- video source
if not options.source then options.source = 'camera' end
source = nn.ImageSource(options.source)

-- retrieve trained network
convnet = nn.Sequential()
file = torch.DiskFile('../trained-nets/network-face-detect', 'r')
convnet:read(file)
file:close()

-- various transformers:
rgb2y = nn.ImageTransform('rgb2y')
rgb2hsl = nn.ImageTransform('rgb2hsl')
scales = {0.3, 0.24, 0.192, 0.15, 0.12, 0.1}
packer = nn.PyramidPacker(scales, convnet)
unpacker = nn.PyramidUnPacker(convnet)

-- for segmentation:
scaler = nn.ImageRescale(320,240,3)

-- displayers to hold print buffers
displayer_source = Displayer()
displayer_features = Displayer()
displayer_preproc = Displayer()
displayer_result = Displayer()
displayer_last = Displayer()

-- profiler
profiler = Profiler()

-- store detections
lastfaces = {}
glbptr = 1

----------------------------------------------------------------------
-- ROUTINES: we use two routines: process() and paint()
--
local function process()
   -- profiler
   profiler:start('full-loop','fps')

   -- get frame from camera, convert to YUV, and just keep Y
   profiler:start('get-frame')
   camFrame = source:forward()
   frameY = rgb2y:forward(camFrame)
   profiler:lap('get-frame')

   -- generate pyramid of scales
   profiler:start('pyramid-pack')
   pyramid, coordinates = packer:forward(frameY)
   profiler:lap('pyramid-pack')

   -- forward prop
   profiler:start('convnet')
   packed_result = convnet:forward(pyramid)
   profiler:lap('convnet')

   -- unpack
   profiler:start('pyramid-unpack')
   outputMaps = unpacker:forward(packed_result, coordinates)
   profiler:lap('pyramid-unpack')

   -- find blobs
   profiler:start('find-blobs')
   listOfFaces = nil
   for i = 1,#outputMaps do
      listOfFaces = image.findBlobs{tensor=outputMaps[i], threshold=0.02*vslide-1, 
                                    maxsize=10, discardClass=2, scale=scales[i],
                                    labels={"face"},
                                    listOfBlobs=listOfFaces}
   end
   profiler:lap('find-blobs')

   -- Extract Centroids of detections
   profiler:start('order-blobs')
   listOfFaces = image.reorderBlobs(listOfFaces)
   listOfFaces = image.remapBlobs(listOfFaces)
   listOfFaces = image.mergeBlobs(listOfFaces, 5)
   profiler:lap('order-blobs')

   -- create a list of seeds, for the segmentation algo
   profiler:start('segmentation')
   local i = 1
   local done = 0
   local seeds = {}
   while true do
      if (listOfFaces[i] ~= nil) then
         local x = listOfFaces[i].x
         local y = listOfFaces[i].y
         local scale = listOfFaces[i].scale

         -- new: extract a patch around the detection, in HSL space,
         --      then compute the histogram of the Hue
         --      for skin tones, the Hue is always < 0.05
         local patch = camFrame:narrow(1,x*4,32):narrow(2,y*4,32)
         local patchH = rgb2hsl:forward(patch):select(3,1)
         local hist = lab.hist(patchH,20)

         -- positives:
         if hist.max.val < 0.05 then
            table.insert(seeds, {x*4+16/scale, y*4+16/scale, 1})
            table.insert(seeds, {x*4+16/scale - 8, y*4+16/scale, 1})
            table.insert(seeds, {x*4+16/scale + 8, y*4+16/scale, 1})
            table.insert(seeds, {x*4+16/scale, y*4+16/scale - 8, 1})
            table.insert(seeds, {x*4+16/scale, y*4+16/scale + 8, 1})
         end
         done = done + 1
      end
      i = i + 1
      if (done == listOfFaces.nbOfBlobs) then
         break
      end
   end
   local smallInput = camFrame --scaler:forward(camFrame)
   table.insert(seeds, {20, 20, 2})
   table.insert(seeds, {20, smallInput:size(2)-20, 2})
   table.insert(seeds, {smallInput:size(1)-20, 20, 2})
   table.insert(seeds, {smallInput:size(1)-20, smallInput:size(2)-20, 2})

   -- compute segmentation
   segments = powerwatersegm.infer{image = smallInput, algo = 1, seeds = seeds}
   profiler:lap('segmentation')
end

local function paint()
   painter:gbegin()
   painter:showpage()
   profiler:start('display')

   -- image from cam
   displayer_source:show{tensor=frameY, painter=painter, 
                         globalzoom=zoom, 
                         min=0, max=1,
                         offset_x=0, offset_y=0,
                         legend='Face detection'}

   -- remove background !!
   profiler:start('bg-suppress')
   frameY:map(segments, function (i,s) if (s==0) then return i else return 0 end end)
   profiler:lap('bg-suppress')
   displayer_source:show{tensor=frameY, painter=painter, 
                         globalzoom=zoom, 
                         min=0, max=1,
                         offset_x=frameY:size(1), offset_y=0,
                         legend='Face detection'}

   -- print last detects
   local off_x = 0
   for i,face in ipairs(lastfaces) do
      displayer_last:show{tensor=face, painter=painter,
                          globalzoom=zoom,
                          min=0, max=1,
                          offset_x=off_x, offset_y=frameY:size(2)+16}
      off_x = off_x + face:size(1)
   end

   -- print boxes
   local i = 1
   local done = 0
   while true do
      if (listOfFaces[i] ~= nil) then
         local x = listOfFaces[i].x
         local y = listOfFaces[i].y
         local scale = listOfFaces[i].scale

         -- new: extract a patch around the detection, in HSL space,
         --      then compute the histogram of the Hue
         --      for skin tones, the Hue is always < 0.05
         local patch = camFrame:narrow(1,x*4,32):narrow(2,y*4,32)
         local patchH = rgb2hsl:forward(patch):select(3,1)
         local hist = lab.hist(patchH,20)

         -- only display skin-tone detections
         if hist.max.val < 0.15 or hist.max.val > 0.9 then
            image.qtdrawbox{ painter=painter,
                             x=x * 4,
                             y=y * 4,
                             w=32/scale,
                             h=32/scale,
                             globalzoom=zoom, 
                             legend=listOfFaces[i].tag}

            -- store face
            if x*4>=1 and y*4>=1 
               and (x*4+32/scale-1)<=frameY:size(1) 
               and (y*4+32/scale-1)<=frameY:size(2) then

               local face = torch.Tensor(32/scale, 32/scale)
               face:copy(frameY:narrow(1,x*4,32/scale):narrow(2,y*4,32/scale))

               lastfaces[glbptr] = nil
               collectgarbage()
               lastfaces[glbptr] = face
               glbptr = glbptr + 1
               if glbptr == 10 then glbptr = 1 end
            end
         end
         done = done + 1
      end
      i = i + 1
      if (done == listOfFaces.nbOfBlobs) then
         break
      end
   end
   profiler:lap('display')
   profiler:lap('full-loop')

   -- disp times
   --profiler:displayAll{painter=painter, x=10, y=frameY:size(2)*1.2*zoom, zoom=zoom}
   painter:gend()
end


function demo()
   -- Loop Process
   local timer = qt.QTimer()
   timer.interval = 1
   timer.singleShot = true
   timer:start()
   qt.connect(timer,'timeout()',
              function() 
                 vslide = widget.verticalSlider.value
                 hslide = widget.horizontalSlider.value
                 zoom = 1/hslide
                 process()
                 if widget.checkBox3.checked then
                    paint()
                 end
                 timer:start()
              end )
   
   -- Close Process   
   widget.windowTitle = "Face Detection"
   widget:show()

end

demo()

