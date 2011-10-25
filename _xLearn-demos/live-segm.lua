#!/usr/bin/env qlua
----------------------------------------------------------------------
-- WHAT: segments a image source.
--       the image source must be provided on the command line.
--
require 'XLearn'
require 'os'
require 'mstsegm'

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
widget = qtuiloader.load('live-segm.ui')
painter = qt.QtLuaPainter(widget.frame)

-- video source
source = nn.ImageSource{type = options.source or 'camera', 
                        path = options.path,
                        cam_idx = options.camidx,
                        fps = 20,
                        width = 200,
                        height = 150}

-- displayers
displayer_source = Displayer()
displayer_segments = Displayer()

-- global zoom
zoom = 2

-- profiler
profiler = Profiler()

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
function run()

   profiler:start('global', 'fps')
   profiler:start('get-frame')
   frame = source:forward()
   profiler:lap('get-frame')

   profiler:start('get-segments')
   segments = segm(frame)
   profiler:lap('get-segments')
   
   painter:gbegin()
   painter:showpage()
   profiler:start('display')
   
   displayer_source:show{tensor = frame:add(segments), painter = painter, globalzoom=zoom, 
                         min=0,max=2, offset_x=0, offset_y=10, 
                         legend='camera image'}
   
   displayer_segments:show{tensor = segments, painter = painter, globalzoom=zoom, 
                           min = 0, max = 1, 
                           offset_x = frame:size(1), offset_y = 10, 
                           legend = 'segmented image'}
   
   profiler:lap('display')
   profiler:lap('global')

   --disp times
   profiler:displayAll{painter = painter, x = 10, y = 420, zoom=1/2}

   -- and params
   minsize = widget.verticalSlider_1.value
   k = widget.verticalSlider_2.value
   painter:setfont(qt.QFont{serif=false,italic=false,size=14})
   painter:moveto(500, 420); painter:show('k = ' .. k)
   painter:moveto(500, 440); painter:show('min = ' .. minsize)

   painter:gend()
end

-- Loop Process
local timer = qt.QTimer()
timer.interval = 5
timer.singleShot = true
timer:start()
qt.connect(timer, 'timeout()', function() run() timer:start() end)

-- Start Process   
widget.windowTitle = "Live Segm"
widget:show()
