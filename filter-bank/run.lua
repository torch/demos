#!/usr/bin/env torch
------------------------------------------------------------
-- a simple filter bank
--
-- Clement Farabet
--

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'qttorch'
require 'camera'
require 'nn'

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

-- threads
torch.setnumthreads(4)

-- filters
filters = nn.SpatialConvolutionMap(nn.tables.random(3,16,1),5,5)

-- profiler
p = xlua.Profiler()

-- process function
function process()
   -- grab frame
   p:start('grab','fps')
   frame = camera:forward()
   p:lap('grab')

   -- transform
   p:start('filter','fps')
   frames = image.scale(frame,320,240)
   transformed = filters(frames)
   p:lap('filter')
end

-- display function
function display()
   p:start('display','fps')
   win:gbegin()
   win:showpage()
   image.display{image=frame, min=0, max=1, win=win, saturate=false}
   image.display{image=transformed, min=-2, max=2, nrow=4,
                 win=win, zoom=1/2, x=frame:size(3), saturate=false}
   win:gend()
   p:lap('display')
end

-- setup gui
local timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              p:start('full loop','fps')
              process()
              display()
              collectgarbage()
              p:lap('full loop')
              p:printAll()
              timer:start()
           end)

widget.windowTitle = 'A random 16-filter bank'
widget:show()
timer:start()
