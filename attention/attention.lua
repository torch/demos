#!/usr/bin/env qlua
------------------------------------------------------------
-- e. culurciello
-- temporal difference simulator

require 'XLearn'
require 'os'
require 'torch'
require 'lab'

-- parse args
op = OptionParser('%prog -s SOURCE [options]')
op:add_option{'-s', '--source', action='store', dest='source', 
              help='image source, can be one of: camera | lena'}
options,args = op:parse_args()

-- need QT
toolBox.useQT()

-- resource file
local ui = paths.thisfile('attention.ui')

-- video source
if not options.source then options.source = 'camera' end
source = nn.ImageSource(options.source)

-- Camera frames
local camFrameA = torch.Tensor(640,480,3)
local camFrameB = torch.Tensor(640,480,3)
local frameY = torch.Tensor(640,480)
local frameY2 = torch.Tensor(640,480)
local frameTD = torch.Tensor(640,480)
--local salience_big = torch.Tensor(640, 480)
local widget = qtuiloader.load(ui)
local painter = qt.QtLuaPainter(widget.frame)

local vslide = 0
local hslide = 0
local nPoints = 10
local pointSize = 32
local tempdiffWeight = .5
local colorThreshold = 0.4

-- get frame from camera
camFrameA:copy( source:forward() )

-- sobel
local kx = lab.new({-1,-2,-1},{0,0,0},{1,2,1})
local ky = lab.new({-1,0,1},{-2,0,2},{-1,0,1})

-- global results
local ringBuffer = {}
local ringBufferSize = 2
local ringBufferP = -1
local resultsReady = false
local results = {}
local speeds
local alternative

-- time
local timeDisplay = os.clock()
local timeDetect = os.clock()
local timeUpdates = os.clock()

--scale (normalize betweeon 0 and 1)
local function scale(frame)
	scaled_frame = frame:add(-frame:min())
	scaled_frame = scaled_frame:div(scaled_frame:max())
	return scaled_frame
end

-- displayers
local disp = {Displayer(), Displayer(), Displayer(), Displayer(), 
              Displayer(), Displayer(), Displayer(), Displayer()}

-----------------------------------------
-- function that receives a 2D tensor "input" 
-- the function returns the location - (r,c) of the 
-- maximum value in the tensor
-----------------------------------------
function find_max_index(input)

   local max_col = lab.max(input)
   local max_val, idx_col = lab.max(max_col, 2)

   local max_row = lab.max(input,2)
   local max_val, idx_row = lab.max(max_row)

   return idx_row[1][1], idx_col[1][1]
end

-- removes an area around point x y in the given tensor. 
-- THIS FUNCTION SLOWS IT DOWN A LOT IF POINTSIZE IS LARGE
function remove(tensor, x, y)
	local w = tensor:size(1)
	local h = tensor:size(2)
	for i=1,pointSize do
		for j=1,pointSize do
			local xx = x-pointSize/2+i
			local yy = y-pointSize/2+j
			if xx < 1 then xx = 1 end
			if yy < 1 then yy = 1 end
			if xx > w then xx = w end
			if yy > h then yy = h end
			tensor[xx][yy]=0
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
	local x_range = (input:size(1))/pointSize
	local y_range = (input:size(2))/pointSize

	section_holder = torch.Tensor(x_range,y_range)
	for i=1,x_range do
		for j=1,y_range do
			local start_x = (i-1)*pointSize+1
			local start_y = (j-1)*pointSize+1
			x, y = find_max_index(input:sub(start_x,start_x+pointSize-1,start_y,start_y+pointSize-1))					
			section_holder[i][j] = input[x+start_x-1][y+start_y-1]
		end
	end
	return section_holder
end

-- get maps in specified scale
local function getMaps(frame1, frame2, scaling)
	local camFrame = torch.Tensor(frame1:size(1)*scaling, frame1:size(2)*scaling, frame1:size(3))
	image.scale(frame1, camFrame, 'simple')
	image.rgb2y(camFrame, frameY)
	image.scale(frame2, camFrame, 'simple')
	image.rgb2y(camFrame, frameY2)
   
   	--get intensity map
	R = camFrame:select(3,1)
	G = camFrame:select(3,2)
	B = camFrame:select(3,3)
	intensity = frameY2
	--get BY map
	Y = (R+G):mul(0.5)
	BY = (B-Y):cdiv(intensity+1) 
	BY = scale(BY)
	image.lower(BY,colorThreshold)
	BY = scale(BY)
	--get RG map
	RG = (R-G):cdiv(intensity+1)
	RG = scale(RG)
	image.lower(RG,colorThreshold)
	RG = scale(RG)
	--get edge map
	gx = image.convolve(intensity,kx, 'valid')
	gy = image.convolve(intensity,ky, 'valid')
	edge = (gx:pow(2)+gy:pow(2)):sqrt()
	edge = scale(edge)
	edge = image.pad2D(edge, 1, 1, 1, 1)
	
   	-- get motion map / temp diff w/ threshold
   	frameTD = frameTD:resizeAs(frameY2):copy(frameY2):mul(-1):add(frameY):select(3,1)
   	frameTD:abs()

   	local thE = vslide/150
   	image.threshold(frameTD, thE, 1, 0, 1)
   
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
	local temp = camFrameA:select(3,1)
	local salience_big = torch.Tensor():resizeAs(temp)
	image.scale(salience, salience_big, 'simple')
	return salience_big
end
	
-- display internal maps
local function displayInternals(camFrame_small, intensity, BY, RG, edge, framTD, fps, offset_X, offset_Y, zoom)
	-- display color scaled image
        disp[1]:show{tensor=camFrame_small, painter=painter,
                   globalzoom=zoom, offset_x=offset_X, offset_y=offset_Y, legend='Input'}
                   
   	--disp temp diff image
   	disp[2]:show{tensor=frameTD, painter=painter,
                   globalzoom=zoom, offset_x=offset_X, offset_y=offset_Y+30+frameY:size(2), legend='Temp Diff'}
                   
   	--disp intensity
   	disp[3]:show{tensor=intensity, painter=painter,
                   globalzoom=zoom, offset_x=offset_X+frameY:size(1), offset_y=offset_Y, legend='Intensity'}

	--disp BY
   	disp[4]:show{tensor=BY, painter=painter,
                   globalzoom=zoom, offset_x=offset_X+frameY:size(1), offset_y=30+offset_Y+frameY:size(2), legend='BY'}

	--disp RG
   	disp[5]:show{tensor=RG, painter=painter,
                   globalzoom=zoom, offset_x=offset_X+2*frameY:size(1), offset_y=offset_Y, legend='RG'}
                   
   	-- disp sobel image
   	disp[6]:show{tensor=edge, painter=painter,
                   globalzoom=zoom, offset_x=offset_X+2*frameY:size(1), offset_y=30+offset_Y+frameY:size(2), legend='Sobel'}
	
	-- disp FPS
   	painter:moveto(offset_X*zoom, (offset_Y+60+RG:size(2)*2)*zoom)
   	painter:show(string.format('FPS = %0f', fps))
end


-- display
local function displayer()
 	
 	local startTime = os.clock()
 	timeDisplay = startTime

 	--get second frame
	camFrameB = source:forward()

	-- paint
   	painter:gbegin()
   	painter:showpage()
   	painter:setcolor("black")
   	painter:setfont(qt.QFont{size=10})
   	local offset_Y = 30
   	local offset_X = 700
	local zoom = 1/hslide

   	-- display input
	disp[7]:show{tensor=camFrameB, painter=painter,
                   globalzoom=zoom, offset_x=0, offset_y=30, legend='CAMERA INPUT, 640x480'}
                   
    --subsample (original is 640 x 480. We subsample to 160 x 120 and 80 x 60)
	local scales = {1/4, 1/8, 1/16}
	-------------------------------------------------------------------------------
	-- 160 x 120 
   	camFrameA_small, intensity1, BY1, RG1, edge1, frameTD1, fps1 = getMaps(camFrameA, camFrameB, scales[1])
   	   	
	--display internals    
	if widget.checkBox1.checked then
		displayInternals(camFrameA_small, intensity1, BY1, RG1, edge1, frameTD1, fps1, offset_X, offset_Y, zoom)
	end
	
	-- create salience map (WEIGHTS ARE ARBITRARY NOW)
	salience1 = createSalienceMap(frameTD1, edge1, RG1, BY1, intensity1)
 	-------------------------------------------------------------------------------
	-- 80 x 60
	camFrameB_small, intensity2, BY2, RG2, edge2, frameTD2, fps2 = getMaps(camFrameA, camFrameB, scales[2])
   	
	-- time
	local diff = os.clock() - timeDisplay
	timeDisplay = os.clock()
	local fps = 1/diff
	
	local offset_Y = 400
	
	--display internals
	if widget.checkBox1.checked then
		displayInternals(camFrameB_small, intensity2, BY2, RG2, edge2, framTD2, fps2, offset_X, offset_Y, zoom)
	end
		
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
	
	--display internals
	if widget.checkBox1.checked then
		displayInternals(camFrame3_small, intensity3, BY3, RG3, edge3, framTD3, fps3, offset_X, offset_Y, zoom)
	end
		
	-- create salience map (WEIGHTS ARE ARBITRARY NOW)
	salience3 = createSalienceMap(frameTD3, edge3, RG3, BY3, intensity3)
	-------------------------------------------------------------------------------
	
	-- combine saliency maps (WEIGHTS ARBITRARY)
	salience_final = (salience1:add(salience2):add(salience3)):div(3)
	if widget.checkBox1.checked then
	-- disp salience map
   		disp[8]:show{tensor=salience_final, painter=painter,
                             min=-1, max=1,
                             globalzoom=zoom, offset_x=0, offset_y=camFrameA:size(2)+90, 
                             legend='FINAL SALIENCE MAP, 640x480'}
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
	    	image.qtdrawbox{ painter=painter,
	                     x=x,
	                     y=y+30,
	                     w=pointSize,
	                     h=pointSize,
	                     globalzoom=zoom, 
	                     legend=i
	                     }	
	    end
	    
    else -- draw red squares
	    saliencetemp = torch.Tensor():resizeAs(salience_final):copy(salience_final)
	    for i=1,nPoints do
	    	local x, y = find_max_index(saliencetemp)
	    	image.qtdrawbox{ painter=painter,
	                     x=x-pointSize/2,
	                     y=y+30-pointSize/2,
	                     w=pointSize,
	                     h=pointSize,
	                     globalzoom=zoom, 
	                     legend=i
	                     }
	         saliencetemp = remove(saliencetemp, x, y)
	    end
    end
    
    -- disp FPS
	local fpsTotal = 1/(os.clock()-startTime)
   	painter:setcolor("black")
   	painter:setfont(qt.QFont{size=10})
   	painter:moveto(10*zoom, (camFrameB:size(2)+60)*zoom)
   	painter:show(string.format('FPS = %0f', fpsTotal))
    
	-- end of paint
	painter:grestore()
	painter:gend()
	
	-- keep current frame
	camFrameA:resizeAs(camFrameB):copy(camFrameB)
end

-- Create QT events
local timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,'timeout()',
           function() 
              vslide = widget.verticalSlider2.value
              hslide = widget.verticalSlider1.value
              alternative = widget.checkBox2.checked
              nPoints = widget.spinBox1.value
              pointSize = widget.spinBox2.value
              tempdiffWeight = widget.doubleSpinBox1.value
              colorThreshold = widget.doubleSpinBox2.value
              displayer()
              timer:start()
           end )

-- Close Process
qt.connect(qt.QtLuaListener(widget),
           'sigShow(bool)',
           function(b) 
              if b then timer:start() end 
           end )

widget.windowTitle = "Temporal Difference Imaging"
widget:show()
