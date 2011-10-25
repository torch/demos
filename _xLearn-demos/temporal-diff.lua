#!/usr/bin/env qlua

-- e. culurciello
-- temporal difference simulator

require 'XLearn'
require 'thread'
require 'os'
local queue = require 'thread.queue'

-- need QT
toolBox.useQT()

-- resource file
local ui = paths.thisfile('face-detect.ui')

-- Camera frames
local camera = nn.ImageSource('camera')
local camFrame = torch.Tensor(640,480,3)
local frameY = torch.Tensor(640,480,1)
local frameY2 = torch.Tensor(640,480,1)
local frameTD = torch.Tensor(640,480,1)
--local frameS = torch.Tensor(640,480,1)
local widget = qtuiloader.load(ui)
local painter = qt.QtLuaPainter(widget.frame)

local vslide = 0
local hslide = 0

-- first frame
camFrame = camera:forward()
image.rgb2y(camFrame, frameY)

-- sobel
local sobel = lab.new({1,1,1},{1,-9,1},{1,1,1})

-- global results
local detectorIsOn
local ringBuffer = {}
local ringBufferSize = 2
local ringBufferP = -1
local resultsReady = false
local results = {}
local speeds

-- multithread mutex
local mutexInput = thread.newmutex()
local mutexResults = thread.newmutex()
local fifo = queue.newqueue(SIZE)

-- time
local timeDisplay = os.clock()
local timeDetect = os.clock()
local timeUpdates = os.clock()

-- displayers
local disp_sob = Displayer()
local disp_in = Displayer()
local disp_temp = Displayer()

-- display
local function displayer(mutexInput, mutexResults)
   -- get another frame
   camFrame = camera:forward()
   image.rgb2y(camFrame, frameY2)
   
   -- temp diff w/ threshold
   frameTD:copy(frameY):mul(-1):add(frameY2)
   local thE = vslide/150
   image.threshold(frameTD, thE)
   
   --sobel operator
   frameS = image.convolve(frameY, sobel, 'valid')

   -- time
   local diff = os.clock() - timeDisplay
   timeDisplay = os.clock()
   local fps = 1/diff

   -- paint
   painter:gbegin()
   painter:showpage()
   
   -- disp cam image
   local zoom = 1/hslide
   disp_in:show{tensor=frameY, painter=painter,
                globalzoom=zoom, offset_x=0, offset_y=0}
                   
   -- disp temp diff image
   disp_temp:show{tensor=frameTD, painter=painter,
                   globalzoom=zoom, offset_x=0, offset_y=frameY:size(2)}
                   
   -- disp sobel image
   disp_sob:show{tensor=frameS, painter=painter,
                 globalzoom=zoom, offset_x=frameY:size(1), offset_y=0}

   -- disp FPS
   painter:moveto(10, frameY:size(2)*2*zoom+30)
   painter:show(string.format('FPS = %0f', fps))


   -- end of paint
   painter:grestore()
   painter:gend()

   -- keep current frame
   frameY:copy(frameY2)

   -- collect garbage
   collectgarbage()
end


-- Create QT events
local timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,'timeout()',
           function() 
              vslide = widget.verticalSlider.value
              hslide = widget.horizontalSlider.value
              detectorIsOn = widget.checkBox3.checked
              displayer(mutexInput, mutexResults)
              timer:start()
           end )

-- Close Process
local listener = qt.QtLuaListener(widget)
qt.connect(listener,'sigClose()',
           function()
              camera:stop()
              timer:stop()
              timer:deleteLater()
              widget:deleteLater()
           end )
qt.connect(listener,'sigShow(bool)',
           function(b) 
              if b then timer:start() end 
           end )

widget.windowTitle = "Temporal Difference Imaging"
widget:show()

