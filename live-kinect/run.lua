#!/usr/bin/env qlua
----------------------------------------------------------------------
-- WHAT: segments a image source.
--       the image source must be provided on the command line.
--
require 'XLearn'
require 'os'
require 'mstsegm'
require 'mincut'
require 'kinect'

-- parse args
op = OptionParser('%prog -s SOURCE [options]')
op:add_option{'-s', '--source', action='store', dest='source', 
              help='image source, can be one of: camera | lena | video'}
op:add_option{'-c', '--camera', action='store', dest='camidx', 
              help='if source=camera, you can specify the camera index: /dev/videoIDX [default=0]'}
op:add_option{'-p', '--path', action='store', dest='path', 
              help='path to video'}
options,args = op:parse_args()

-- setup QT gui
toolBox.useQT()
widget = qtuiloader.load('live-kinect.ui')
painter = qt.QtLuaPainter(widget.frame)

-- video source
-- source = nn.ImageSource{type = options.source or 'camera', 
--                         path = options.path,
--                         cam_idx = options.camidx,
--                         fps = 20,
--                         width = 200,
--                         height = 150}

-- displayers
displayer_source = Displayer()
displayer_depth = Displayer()
displayer_segments = Displayer()

-- global zoom
zoom = 1
kinect = kinect.Device(640,480)
-- incremental segmentation
do
   local nslices = 8
   local segmbuf
   local ringbuf = {}
   local ringbuf_i = 0

   -- incremental segm function
   function segm(img)
      -- new frame
      ringbuf_i = ringbuf_i + 1

      -- resize buffer
      segmbuf = segmbuf or torch.Tensor(img:size(1), img:size(2), img:size(3), nslices)
      -- store frame
      ringbuf[ringbuf_i] = torch.Tensor():resizeAs(img):copy(img)

      -- cleanup
      ringbuf[ringbuf_i-nslices] = nil
      collectgarbage()

      -- compute segm
      local segments
      if ringbuf_i > nslices then
         -- fill buffer
         for i = 1,nslices do
            segmbuf:select(4,i):copy(ringbuf[ringbuf_i-nslices+i])
         end
         -- segm
         segments = mstsegm.infer{image=segmbuf, 
                                  connex=4, 
                                  k=k, min=minsize,
                                  incremental=true}
         -- return last slice
         return segments:select(4,segments:size(4))
      else
         -- in the mean time, dont do anything
         return img
      end
   end
end

-- exec

input = torch.Tensor(320,240,4)
result = torch.Tensor()
function run()
   -- get stream
   kinect:getRGBD()
   kinect:tilt(widget.verticalSlider2.value)
   image.scale(kinect.rgbd,input,'simple')
   frame = input:narrow(3,1,3)
   thre = widget.verticalSlider_1.value

   mincut.segmentation{image = input,
                       result=result,
                       threshold=thre}
   
   painter:gbegin()
   painter:showpage()
   
   displayer_source:show{tensor = frame, painter = painter, globalzoom=zoom, 
                         min=0,max=1, offset_x=0, offset_y=10, 
                         legend='camera image'}
   displayer_depth:show{tensor = input:select(3,4), painter = painter, globalzoom=zoom, 
                        min = 0, max = 1, 
                        offset_x = frame:size(1)+5, offset_y = 10, 
                        legend = 'depth image'}
   displayer_segments:show{tensor = result, painter = painter, globalzoom=zoom, 
                           min = 0, max = 1, 
                           offset_x = 0, offset_y = frame:size(2) + 20, 
                           legend = 'segmented image'}
   
   -- and params
   painter:setfont(qt.QFont{serif=false,italic=false,size=14})
   painter:moveto(330, 460); painter:show('Threshold = ' .. thre)
   painter:gend()
end

-- Loop Process
local timer = qt.QTimer()
timer.interval = 5
timer.singleShot = true
timer:start()
qt.connect(timer, 'timeout()', function() run() timer:start() end)

-- Start Process   
widget.windowTitle = "Live Segmentation on Kinect"
widget:show()
