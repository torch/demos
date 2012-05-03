#!/usr/bin/env torch
------------------------------------------------------------
-- a simple frame grabber demo.
--
-- Clement Farabet
--

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='camera index: /dev/videoIDX', default=0}
opt,args = op:parse()

-- setup GUI (external UI file)
widget = qtuiloader.load('g.ui')
win = qt.QtLuaPainter(widget.frame)

-- setup camera
camera = image.Camera(opt.camidx)

-- process function
function process()
   -- grab frame
   frame = camera:forward()

   -- transform it
   if transform == 'rgb' then 
      transformed = frame
   else
      transformed = image[transform](frame)
   end
end

-- display function
function display()
   zoom = 1
   win:gbegin()
   win:showpage()
   image.display{image=frame, win=win, zoom=zoom}
   image.display{image=transformed, win=win, x=frame:size(3)*zoom, zoom=zoom}
   win:gend()
end

-- setup gui
local timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              process()
              display()
              timer:start()
           end)

-- calls back for all buttons
transform = 'rgb'
qt.connect(qt.QtLuaListener(widget.rgb),
           'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
           function (...) transform = 'rgb' end)

qt.connect(qt.QtLuaListener(widget.rgb2y),
           'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
           function (...) transform = 'rgb2y' end)

qt.connect(qt.QtLuaListener(widget.rgb2yuv),
           'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
           function (...) transform = 'rgb2yuv' end)

qt.connect(qt.QtLuaListener(widget.rgb2hsl),
           'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
           function (...) transform = 'rgb2hsl' end)

widget.windowTitle = 'A Simple Frame Grabber'
widget:show()
timer:start()
