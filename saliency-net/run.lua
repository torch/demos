#!/usr/bin/env qlua
------------------------------------------------------------
-- a simple convnet saliency demo
-- takes output of convnet 1st layer and displays it

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'inline'
require 'camera'
require 'nnx'
require 'openmp'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, you can specify the camera index: /dev/videoIDX', 
          default=0}
op:option{'-n', '--network', action='store', dest='network', 
          help='path to existing [trained] network',
          default='../face-detector/face.net'}
opt,args = op:parse()

torch.setdefaulttensortype('torch.FloatTensor')

-- profiler
p = xlua.Profiler()

-- setup GUI (external UI file)
if not win or not widget then 
   widget = qtuiloader.load('g.ui')
   win = qt.QtLuaPainter(widget.frame) 
end

-- setup camera
camera = image.Camera(opt.camidx)
S_x = 160
S_y = 120
img1=torch.Tensor(3,S_y,S_x)
img2=torch.Tensor(3,S_y,S_x)
frame=torch.Tensor(S_y-4,S_x-4) -- filter is 5x5 so lose 2 pix on each sides

-- load network and clone 1st layer
network = torch.load(opt.network)
network = torch.load(opt.network):float()
net2 = nn.Sequential()
net2:add(network.modules[1])
net2:add(network.modules[2])
--net2:add(network.modules[3])
--net2:add(network.modules[4])

-- process function
function process()
   frame=torch.zeros(S_y-4,S_x-4)
   img2=img2:copy(img1)
   img1 = image.scale(camera:forward(),S_x,S_y,'simple')
   out = net2:forward(img1[2]) -- pass only G map in RGB
   for i=1,8 do
       frame:add(out[i])
   end
   frame:add(-torch.min(frame)):div(torch.max(frame)):mul(0.2) -- normalize
   colorop =  image.crop((img1[2]-img1[1]), 2,2, S_x-2,S_y-2) +
       image.crop((img1[2]-img1[3]), 2,2, S_x-2,S_y-2) -- color opponency
   colorop:add(-torch.min(colorop)):div(torch.max(colorop)):mul(0.2) -- normalize
   tmpdiff = image.crop((img1[2]-img2[2]), 2,2, S_x-2,S_y-2) -- temp diff
   tmpdiff:add(-torch.min(tmpdiff)):div(torch.max(tmpdiff)):mul(0.6) -- normalize
   frame:add(tmpdiff):add(colorop) -- add temp diff, color opp
   frame = image.scale(frame,640,480,'simple')
end   

-- display function
function display()
   win:gbegin()
   win:showpage()
   -- (1) display input image
   image.display{image=frame, win=win, saturation=false, min=0, max=1}
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
              process()
              p:start('display','fps')
              display()
              p:lap('display')
              timer:start()
              p:lap('full loop')
              p:printAll()
           end)
widget.windowTitle = 'Saliency simulator'
widget:show()
timer:start()
