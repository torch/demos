#!/usr/bin/env qlua
------------------------------------------------------------
-- elab faces+attention demo
-- Faye Zhao <faye.zhao@yale.edu>, Ifigeneia Derekli <ifigeneia1989@gmail.com>
-- with Eugenio Culurciello, Polina Akselrod

require 'XLearn'
require 'thread'
require 'os'
local queue = require 'thread.queue'

-- need QT
toolBox.useQT()

-- resource file
local ui = paths.thisfile('faces-attention.ui')

-- Camera frames
local camera = nn.ImageSource('camera')
local camFrame1 = torch.Tensor(640,480,3)
local camFrame2 = torch.Tensor(640,480,3)
local frameY = torch.Tensor(640,480,1)
local frameY1 = torch.Tensor(640,480)
local frameY2 = torch.Tensor(640,480)
local frameTD = torch.Tensor(640,480)
local widget = qtuiloader.load(ui)
local painter = qt.QtLuaPainter(widget.frame)
local salience_final = torch.Tensor(640,480)

local vslide = 0
local hslide = 0
local nPoints = 10
local pointSize = 32
local tempdiffWeight = .5
local colorThreshold = 0.4

local saliencetemp = torch.Tensor(640/pointSize, 480/pointSize)

-- sobel
local kx = lab.new({-1,-2,-1},{0,0,0},{1,2,1})
local ky = lab.new({-1,0,1},{-2,0,2},{-1,0,1})

-- global results
local detectorIsOn
local facesIsOn
local attentionIsOn
local ringBuffer = {}
local ringBufferSize = 2
local ringBufferP = -1
local resultsReady = false
local results = {}
local speeds

-- multithread mutex
local mutexInput = thread.newmutex()
local mutexResults = thread.newmutex()
local fifo = queue.newqueue(SIZE)        -----------?????????????????

-- time
local timeDisplay = os.clock()
local timeDetect = os.clock()
local timeUpdates = os.clock()



-- attention functions

--scale (normalize betweeon 0 and 1)
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

   local max_col = lab.max(input)
   local max_val, idx_col = lab.max(max_col, 2)

   local max_row = lab.max(input,2)
   local max_val, idx_row = lab.max(max_row)

   return idx_row[1][1], idx_col[1][1]
end

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
	image.rgb2y(camFrame, frameY1)
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
   	frameTD = (frameY2-frameY1):select(3,1)
   	frameTD:abs()
   	local thE = vslide/150
   	image.threshold(frameTD, thE, 0, 0, 1)
   	
   
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
	local temp = camFrame1:select(3,1)
	local salience_big = torch.Tensor():resizeAs(temp)
	image.scale(salience, salience_big, 'simple')
	return salience_big
end


-- detector
local function detector(mutexInput, mutexResults)
   -- retrieve trained network
   local convnet = nn.Sequential()
   local file = torch.DiskFile('../trained-nets/network-face-detect', 'r')
   convnet:read(file)
   file:close()

   -- run the network on incoming frames
   while true do
   		         ------------------ DO ATTENTION ---------------------
      	if detectorIsOn and attentionIsOn then
	        --get second frame
                        camFrame2 = camera:forward()
	 		local scales = {1/4, 1/8, 1/16}
			offset_Y = 30
			offset_X = 700
		
			---------------------------160 x 120 PROCESSING-------------------------------
	   		camFrame1_small, intensity1, BY1, RG1, edge1, frameTD1, fps1 = getMaps(camFrame1, camFrame2, scales[1])
	   	   	salience1 = createSalienceMap(frameTD1, edge1, RG1, BY1, intensity1)
	 	
	 		--------------------------80 x 60 PROCESSING--------------------------------
			camFrame2_small, intensity2, BY2, RG2, edge2, frameTD2, fps2 = getMaps(camFrame1, camFrame2, scales[2])
		 	offset_Y = 400
			salience2 = createSalienceMap(frameTD2, edge2, RG2, BY2, intensity2)
		
			-------------------------40 x 30 PROCESSING---------------------------------
			camFrame3_small, intensity3, BY3, RG3, edge3, frameTD3, fps3 = getMaps(camFrame1, camFrame2, scales[3])
	    	offset_Y = 700
			salience3 = createSalienceMap(frameTD3, edge3, RG3, BY3, intensity3)
		
			-------------------FINAL SALIENCY MAP----------------------
			-- combine saliency maps (WEIGHTS ARBITRARY)
			salience_final = (salience1:add(salience2):add(salience3)):div(3)
		    	
			-- keep current frame
			camFrame1:copy(camFrame2)
	
	      	saliencetemp = return_section(salience_final)
	    
         end
         -----------------------------------------------------
         
      if detectorIsOn and facesIsOn then 
      	 
	         -- copy input
	         mutexInput:lock()
	         local inputY = torch.Tensor():resizeAs(frameY):copy(frameY)
	         mutexInput:unlock()
	
	         -- make pyramid 
	         local scales = {1/3, 1/4, 1/5, 1/7}
	         local pyramid = image.makePyramid{tensor=inputY, scales=scales}
	
	         -- forward prop
	         local resultsLocal = nil
	         local outputMaps = {}
	         local i = 1
	         while true do
	            local img = table.remove(pyramid, 1)
	            if img == nil then break end
	            local result = convnet:forward(img)
	            local resulttemp = torch.Tensor( result:size() ):copy(result)
	            table.insert(outputMaps, resulttemp)
	            -- process output map
	            resultsLocal = image.findBlobs{tensor=result, threshold=0.02*vslide-1, 
	                                           discardClass=2, scale=scales[i],
	                                           listOfBlobs=resultsLocal}
	            i=i+1
	         end
	
	         -- time elapsed
	         local diff = os.clock() - timeDetect
	         timeDetect = os.clock()
	
	         -- Process Results
	         mutexResults:lock()
	         -- Extract Centroids of detections
	         resultsLocal = image.reorderBlobs(resultsLocal)
	         resultsLocal = image.remapBlobs(resultsLocal)
	         resultsLocal = image.mergeBlobs(resultsLocal, 50)
	         local ringBufferPprev = ringBufferP
	         ringBufferP = (ringBufferP + 1) % ringBufferSize
	         ringBuffer[ringBufferP] = resultsLocal
	         resultsReady = true
	
	         -- Estimate Speed (1st order)
	         if (ringBufferPprev ~= -1) then
	            speeds = image.getSpeed(ringBuffer[ringBufferP], ringBuffer[ringBufferPprev], diff, 50)
	         end
	  
         

         
         --if detectorIsOn and facesIsOn then
         mutexResults:unlock()
		 --end
      end
      libxlearn.usleep(10000)
   end
end

-- display
inpDisp = Displayer()
local function displayer(mutexInput, mutexResults)
   -- get frame from camera, convert to Y
   camFrame1 = camera:forward()
   mutexInput:lock()
   image.rgb2y(camFrame1, frameY)
   mutexInput:unlock()
   
   -- time
   local diff = os.clock() - timeDisplay
   timeDisplay = os.clock()
   local fps = 1/diff

   -- paint
   painter:gbegin()
   painter:showpage()
   	painter:setcolor("black")
   	painter:setfont(qt.QFont{size=10})

   -- disp cam image
   local zoom = 1/hslide
   inpDisp:show{tensor=camFrame1, painter=painter, min=0, max=1,
                globalzoom=zoom, offset_x=0, offset_y=0}

		
   -- disp FPS
   painter:moveto(10, frameY:size(2)*zoom+30)
   painter:show(string.format('FPS = %0f', fps))

   -- get results from detector
   mutexResults:lock()
   if resultsReady then -- display results
      results = ringBuffer[ringBufferP]
      timeUpdates = os.clock()
      -- disp them
      for i = 1,#results do
         image.qtdrawbox{ painter=painter,
                          x=results[i].x * 4, y=results[i].y * 4,
                          w=32/results[i].scale, h=32/results[i].scale,
                          globalzoom=zoom, 
                          legend='face' }
      end
      resultsReady = false
      
   elseif (speeds ~= nil) then -- display interpolated results
      local interpPos = image.getNextPos(results, speeds, os.clock() - timeUpdates)
      for i = 1,#interpPos do
         image.qtdrawbox{ painter=painter,
                          x=interpPos[i].x * 4, y=interpPos[i].y * 4,
                          w=32/interpPos[i].scale, h=32/interpPos[i].scale,
                          globalzoom=zoom, 
                          legend='face' }
      end
   end
   
   -- disp ATTENTION
	if detectorIsOn and attentionIsOn then 
		local offset_Y = 30
	   	local offset_X = 700
	
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
	end
	painter:setcolor("black")
   	painter:setfont(qt.QFont{size=10})

	-- disp salience map
	if widget.checkBox1.checked then
		image.qtdisplay{tensor=salience_final, painter=painter, raw=true,
   	               globalzoom=zoom, offset_x=0, offset_y=camFrame1:size(2)+120, legend='FINAL SALIENCE MAP, 640x480'}
   	end
	    	
    
    
   mutexResults:unlock()

   -- end of paint
   painter:grestore()
   painter:gend()

   libxlearn.usleep(10000)
end

-- Background processing thread
thread.newthread(detector, {mutexInput, mutexResults})

-- Create QT events
local timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,'timeout()',
           function() 
              vslide = widget.verticalSlider2.value
              hslide = widget.verticalSlider1.value
              nPoints = widget.spinBox1.value
              pointSize = widget.spinBox2.value
              tempdiffWeight = widget.doubleSpinBox1.value
              colorThreshold = widget.doubleSpinBox2.value
			  detectorIsOn = widget.checkBox2.checked
			  facesIsOn = widget.checkBox4.checked
			  attentionIsOn = widget.checkBox3.checked
              displayer(mutexInput, mutexResults)
              timer:start()
          end )

-- Close Process
local listener = qt.QtLuaListener(widget)
qt.connect(listener,'sigClose()',
           function()
              timer:stop()
              timer:deleteLater()
              widget:deleteLater()
           end )
qt.connect(listener,'sigShow(bool)',
           function(b) 
              if b then timer:start() end 
           end )

widget.windowTitle = "Live Processing"
widget:show()

