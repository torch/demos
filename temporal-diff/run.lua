#!/usr/bin/env qlua
------------------------------------------------------------
-- a simple temporal difference demo
--

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
useOpenCV = true
require 'camera'

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
img1=torch.Tensor(3,480,640)
img2=torch.Tensor(3,480,640)

-- process function
function process()
   -- flow - from testme function:
   -- grab frames
   img2=img2:copy(img1)
   img1 = camera:forward()
   --img1 = img1:copy(camera:forward())
   frame = img1-img2  
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

widget.windowTitle = 'Temporal Difference live'
widget:show()
timer:start()
