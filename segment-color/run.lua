#!/usr/bin/env qlua
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
xrequire('camera',true)

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, you can specify the camera index: /dev/videoIDX', 
          default=0}
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
   
   -- color segment:
   segnum = 4 -- since image is max 1 we get 4 colors/plane this way
   frame=torch.ceil(frame*segnum)

end

-- display function
function display()
   zoom = 1
   win:gbegin()
   win:showpage()
   image.display{image=frame, win=win, zoom=zoom}
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

widget.windowTitle = 'Color segmenter'
widget:show()
timer:start()
