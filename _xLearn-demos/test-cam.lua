#!/usr/bin/env qlua
----------------------------------------------------------------------
-- WHAT: executes a pretrained neural net on an image source.
--       the image source must be provided on the command line.
--
require 'XLearn'
require 'os'

-- parse args
op = OptionParser('%prog -s SOURCE [options]')
op:add_option{'-t', '--type', action='store', dest='type', 
              help='optional type, depends on the kind of source. for camera: opencv | camiface | v4linux'}
op:add_option{'-c', '--camera', action='store', dest='camidx', 
              help='if source=camera, you can specify the camera index: /dev/videoIDX [default=0]'}
options,args = op:parse_args()

-- setup QT gui
toolBox.useQT()
painter = qtwidget.newwindow(800,600)

-- video source
source = nn.ImageSource('camera', options.type, options.camidx)

-- displayers to hold print buffers
displayer_source = Displayer()
displayer_chanels = Displayer()

-- global zoom
zoom = 2/3

-- profiler
profiler = Profiler()

local function loop()
   profiler:start('global','fps')

   profiler:start('get-frame')
   frame = source:forward()
   profiler:lap('get-frame')
   
   --frame = image.yuv2rgb(frame)

   painter:gbegin()
   painter:showpage()
   profiler:start('display')

   -- image from cam
   displayer_source:show{tensor=frame, painter=painter, 
                         globalzoom=zoom, 
                         min=0, max=1,
                         inplace=true,
                         offset_x=0, offset_y=0,
                         legend='From Camera'}

   profiler:lap('display')
   profiler:lap('global')

   -- disp times
   profiler:displayAll{painter=painter, x=10, y=frame:size(2)*zoom+100, zoom=1/2}
   painter:gend()
end

-- Loop Process
local timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
timer:start()
qt.connect(timer,'timeout()', function() loop() timer:start() end)
