#!/usr/local/bin/qlua
--------------------------------------------------------------------------------
-- Temporal difference simulator
--------------------------------------------------------------------------------
-- Original author: Aysegul Dundar
-- Updated by: Alfredo Canziani
--------------------------------------------------------------------------------

require 'qtwidget'
require 'qtuiloader'
require 'nn'
require 'camera'

function lowerthreshold (tensor, threshold)
   local t = tensor:contiguous()
   local data = torch.data(t);
   local ydim = t:size(1)
   local xdim = t:size(2)
   for i=0,ydim do
      for j=0,xdim do
         local val = data[ydim*i + xdim]
         if val < -threshold then
            val = -threshold
         elseif  val > threshold then
            val = threshold
         end
      end
   end
   tensor:copy(t)
end

function imagethreshold (tensor, threshold)
   local t = tensor:contiguous()
   local data = torch.data(t);
   local ydim = t:size(1)
   local xdim = t:size(2)
   for i=0,ydim do
      for j=0,xdim do
         local val = data[ydim*i + xdim]
         if val < threshold then
            val = 0
         else
            val = val - threshold
         end
      end
   end
   tensor:copy(t)
end

-- setup GUI (external UI file)
widget = qtuiloader.load('attention.ui')
win = qt.QtLuaPainter(widget.frame)

-- Camera frames
local camFrameA = torch.Tensor(3, 480, 640)
local camFrameB = torch.Tensor(3, 480, 640)

local vslide = 0
local nPoints = 10
local pointSize = 32
local tempdiffWeight = 0.5
local colorThreshold = 0.4

-- get frame from camera
camera = image.Camera{}
camFrameA = camFrameA:copy(camera:forward())
--camFrameA = camera:forward()
-- sobel
local kx= torch.Tensor(3,3)
s=kx:storage()
s[1]=-1 s[2]=-2 s[3]=-1
s[4]=0 s[5]=0 s[6]=0
s[7]=1 s[8]=2 s[9]=1

local ky= torch.Tensor(3,3)
n=ky:storage()
n[1]=-1 n[2]=0 n[3]=1
n[4]=-2 n[5]=0 n[6]=2
n[7]=-1 n[8]=0 n[9]=1

-- global results
--local alternative

-- time
local timeDisplay = os.clock()
local timeDetect = os.clock()
local timeUpdates = os.clock()

--scale (normalize between 0 and 1)
local function scale(frame)
   scaled_frame = frame:add(-frame:min())
   scaled_frame = scaled_frame:div(scaled_frame:max())
   return scaled_frame
end

-----------------------------------------
-- function that receives a 2D tensor "input"
-- the function returns the location - (r,c) of the
-- maximum value in the tensor
-----------------------------------------
function find_max_index(input)

   local max_col = torch.max(input,1)
   local max_val, idx_col = torch.max(max_col, 2)

   local max_row =torch.max(input,2)
   local max_val, idx_row = torch.max(max_row,1)

   return idx_row[1][1], idx_col[1][1]
end

-------------------------------------------------------
-- removes an area around point x y in the given tensor.
-- THIS FUNCTION SLOWS IT DOWN A LOT IF POINTSIZE IS LARGE
---------------------------------------------------------

function remove(tensor, x, y)
   local h = tensor:size(1)
   local w = tensor:size(2)
   for i=1,pointSize do
      for j=1,pointSize do
         local yy = y-pointSize/2+i
         local xx = x-pointSize/2+j
         if xx < 1 then xx = 1 end
         if yy < 1 then yy = 1 end
         if xx > w then xx = w end
         if yy > h then yy = h end
         tensor[yy][xx]=0
      end
   end
   return tensor
end

----------------------------------
--alternative method for finding max: divide image into a grid
--find global maxima within subdivisions
--nPoints number of grids
-----------------------------------

function return_section(input)
   local y_range = (input:size(1))/pointSize
   local x_range = (input:size(2))/pointSize

   section_holder = torch.Tensor(y_range,x_range)
   for i=1,y_range do
      for j=1,x_range do
         local start_y = (i-1)*pointSize+1
         local start_x = (j-1)*pointSize+1
         x, y = find_max_index(input:sub(start_y,start_y+pointSize-1,start_x,start_x+pointSize-1))
         section_holder[i][j] = input[y+start_y-1][x+start_x-1]
      end
   end
   return section_holder
end

-- get maps in specified scale
local function getMaps(frame1, frame2, scaling)
   local camFrame = torch.Tensor(frame1:size(1), frame1:size(2)*scaling, frame1:size(3)*scaling)

   image.scale(camFrame, frame1, 'simple')
   frameY = image.rgb2y(camFrame)

   image.scale(camFrame, frame2, 'simple')
   frameY2 = image.rgb2y(camFrame)

   --get intensity map

   R = camFrame:select(1,1)
   G = camFrame:select(1,2)
   B = camFrame:select(1,3)

   intensity=frameY2[1]

   --get BY map
   Y = (R+G):mul(0.5)
   BY = (B-Y):cdiv(intensity+1)
   BY = scale(BY)
   lowerthreshold(BY,colorThreshold)
   BY = scale(BY)

   --get RG map
   RG = (R-G):cdiv(intensity+1)
   RG = scale(RG)
   lowerthreshold(RG,colorThreshold)
   RG = scale(RG)

   --get edge map

   gx = image.convolve(intensity,kx, 'valid')
   gy = image.convolve(intensity,ky, 'valid')
   edge = (gx:pow(2)+gy:pow(2)):sqrt()

   edge = scale(edge)
   padder = nn.SpatialZeroPadding(1,1,1,1)
   edge = torch.Tensor(edge):resize(1, edge:size(1), edge:size(2))
   edge=padder:forward(edge)
   edge = edge[1]
   -- get motion map / temp diff w/ threshold

   frameTD = torch.Tensor():resizeAs(frameY2):copy(frameY2):mul(-1):add(frameY):select(1,1)

   frameTD:abs()
   local thE = vslide/150
   imagethreshold(frameTD, thE)

   -- time
   local diff = os.clock() - timeDisplay
   timeDisplay = os.clock()
   local fps = 1/diff

   return camFrame, intensity, BY, RG, edge, frameTD, fps

end

-- create salience map (WEIGHTS ARE ARBITRARY NOW)
local function createSalienceMap(frameTD, edge, RG, BY, intensity)

   local edgeWeight = (1-tempdiffWeight)/4
   local rgWeight = (1-tempdiffWeight)/4
   local byWeight = (1-tempdiffWeight)/4
   local intensityWeight=(1-tempdiffWeight)/4

   salience = frameTD:mul(tempdiffWeight):add(edge:mul(edgeWeight)):add(RG:mul(rgWeight)):add(BY:mul(byWeight)):add(intensity:mul(intensityWeight))

   -- back to 640 x 480
   local temp = camFrameA[1]

   local salience_big = torch.Tensor():resizeAs(temp)
   image.scale(salience_big, salience, 'simple')
   return salience_big
end

-- display internal maps
local function displayInternals(camFrame_small, intensity, BY, RG, edge, frameTD, fps, offset_X, offset_Y, zoom )

   -- display color scaled image
   image.display{image=camFrame_small, win=win,zoom=zoom,
   x=offset_X, y=offset_Y, legend='Input'}

   win:moveto(offset_X, 20)
   win:setfont(qt.QFont{size=14})
   win:show(string.format('Input'))

   --disp intensity
   image.display{image=intensity, win=win, zoom=zoom,
   x=offset_X,y=offset_Y+30+camFrame_small:size(2), legend='Intensity'}

   win:moveto(offset_X, 20+30+camFrame_small:size(2))
   win:show(string.format('Intensity'))

   --disp BY
   image.display{image=BY, win=win, zoom=zoom,
   x=camFrame_small:size(3)+offset_X, y=offset_Y, legend='BY'}

   win:moveto(camFrame_small:size(3)+offset_X, 20)
   win:show(string.format('BY'))

   --disp RG
   image.display{image=RG, win=win,zoom=zoom,
   x=camFrame_small:size(3)+offset_X, y=offset_Y+30+camFrame_small:size(2), legend='RG'}
   win:moveto(camFrame_small:size(3)+offset_X, 20+30+camFrame_small:size(2))
   win:show(string.format('RG'))

   -- disp sobel image
   image.display{image=edge,  win=win,  zoom=zoom,
   x=2*camFrame_small:size(3)+offset_X, y=offset_Y, legend='Sobel'}
   win:moveto(2*camFrame_small:size(3)+offset_X, 20)
   win:show(string.format('Sobel'))

   --disp temp diff image
   image.display{image=frameTD, win=win,  zoom=zoom,
   x=2*camFrame_small:size(3)+offset_X, y=offset_Y+30+camFrame_small:size(2), legend='Temp Diff'}
   win:moveto(2*camFrame_small:size(3)+offset_X, 20+30+camFrame_small:size(2))
   win:show(string.format('Temp Diff'))


   -- disp FPS

   --win:moveto(offset_X*zoom, (offset_Y+60+RG:size(1)*2)*zoom)
   --win:show(string.format('FPS = %0f', fps))

end

-- display
local function displayer()

   local startTime = os.clock()
   timeDisplay = startTime

   --get second frame

   camFrameB = camera:forward()

   win:gbegin()
   win:showpage()

   local offset_Y = 30
   local offset_X = 700
   local zoom = 1

   -- display input
   image.display{image=camFrameB, win=win, zoom=1, x=0, y=30, legend='CAMERA INPUT, 640x480'}
   win:moveto(camFrameA:size(2)/2, 20)
   win:setfont(qt.QFont{size=14})
   win:show(string.format('CAMERA INPUT, 640x480'))

   --subsample (original is 640 x 480. We subsample to 160 x 120 and 80 x 60)
   local scales = {1/4, 1/8, 1/16}
   -------------------------------------------------------------------------------
   -- 160 x 120
   camFrame1_small, intensity1, BY1, RG1, edge1, frameTD1, fps1 = getMaps(camFrameA, camFrameB, scales[1])

   --display internals
   if widget.checkBox1.checked then
      displayInternals(camFrame1_small, intensity1, BY1, RG1, edge1, frameTD1, fps1, offset_X, offset_Y, zoom)

   end

   -- create salience map (WEIGHTS ARE ARBITRARY NOW)
   salience1 = createSalienceMap(frameTD1, edge1, RG1, BY1, intensity1)
   -------------------------------------------------------------------------------
   -- 80 x 60
   camFrame2_small, intensity2, BY2, RG2, edge2, frameTD2, fps2 = getMaps(camFrameA, camFrameB, scales[2])

   -- time
   local diff = os.clock() - timeDisplay
   timeDisplay = os.clock()
   local fps = 1/diff

   local offset_Y = 400

   -- create salience map (WEIGHTS ARE ARBITRARY NOW)
   salience2 = createSalienceMap(frameTD2, edge2, RG2, BY2, intensity2)
   -------------------------------------------------------------------------------
   -- 40 x 30
   camFrame3_small, intensity3, BY3, RG3, edge3, frameTD3, fps3 = getMaps(camFrameA, camFrameB, scales[3])

   -- time
   local diff = os.clock() - timeDisplay
   timeDisplay = os.clock()
   local fps = 1/diff

   local offset_Y = 700

   -- create salience map (WEIGHTS ARE ARBITRARY NOW)
   salience3 = createSalienceMap(frameTD3, edge3, RG3, BY3, intensity3)
   -------------------------------------------------------------------------------

   -- combine saliency maps (WEIGHTS ARBITRARY)
   salience_final = (salience1:add(salience2):add(salience3)):div(3)
   if widget.checkBox1.checked then
      -- disp salience map
      image.display{image=salience_final,win=win, zoom=zoom/2, x=700, y=350,
      legend='FINAL SALIENCE MAP, 640x480'}

      win:moveto(700, 340)
      win:show(string.format('FINAL SALIENCE MAP, 640x480'))
   end

   --draw squares
   if (alternative) then
      --- alternative draw squares
      saliencetemp = return_section(salience_final)
      for i=1,nPoints do
         local x, y = find_max_index(saliencetemp)
         saliencetemp[x][y]=0
         x = 1+pointSize*(x-1)
         y = 1+pointSize*(y-1)
         win:rectangle(y, x+30, pointSize, pointSize)
         win:stroke()
      end
   else -- draw red squares
      saliencetemp = torch.Tensor():resizeAs(salience_final):copy(salience_final)
      for i=1,nPoints do
         local x, y = find_max_index(saliencetemp)
         win:rectangle(y-pointSize/2, x+30-pointSize/2, pointSize, pointSize)
         win:stroke()
         saliencetemp = remove(saliencetemp, x, y, pointSize)
      end
   end

   -- disp FPS
   local fpsTotal = 1/(os.clock()-startTime)

   win:setcolor("black")
   win:moveto(10*zoom, (camFrameB:size(2)+60)*zoom)
   win:show(string.format('FPS = %0f', fpsTotal))
   --end of paint
   win:grestore()
   win:gend()
   -- keep current frame
   camFrameA:resizeAs(camFrameB):copy(camFrameB)
end

-- Create QT events
local timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
qt.connect(timer,'timeout()',
function()
   vslide = widget.verticalSlider2.value
   --hslide = widget.verticalSlider1.value
   alternative = widget.checkBox2.checked
   nPoints = widget.spinBox1.value
   pointSize = widget.spinBox2.value
   tempdiffWeight = widget.doubleSpinBox1.value
   colorThreshold = widget.doubleSpinBox2.value
   displayer()
   timer:start()
end )

-- Close Process
widget.windowTitle = "Temporal Difference Imaging"
widget:show()
timer:start()
