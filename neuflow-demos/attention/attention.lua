---------------------------------------------------------------


-- Attention detector
-- The image is grabbed from a webcam and you can run the program both on Neuflow and CPU. 
--
-- Author: Aysegul Dundar
-- Date: 01/18/2012


require 'neuflow'
require 'xlua' 
require 'os'
require 'torch'
require 'qt'
--require 'lab'

require 'qtwidget'
require 'qtuiloader'
xrequire('nnx',true)
xrequire('camera',true)
xrequire('inline',true)

----------------------------------------------------------------------
-- ARGS: parse user arguments

op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, you can specify the camera index: /dev/videoIDX', 
          default=0}
opt,args = op:parse()
target = 'neuflow' or 'cpu'
last_name=target
last_fps = -1
counting = 0
----------------------------------------------------------------------
-- INIT: initialize the neuFlow context
-- a mem manager, the dataflow core, and the compiler


neuFlow = neuflow.init()

----------------------------------------------------------------------
-- ELABORATION: describe the algorithm to be run on neuFlow, and 
-- how it should interact with the host (data exchange)
-- note: any copy**Host() inserted here needs to be matched by
-- a copy**Dev() in the EXEC section.
--


-- image sizes

size = 380
inputs = torch.Tensor(3,size,size)
intensity = torch.Tensor(size,size)
intensity1=torch.Tensor(size,size)
intensitydiv = torch.Tensor(1,size,size)
intermediate = torch.Tensor(4,size,size)
output = torch.Tensor(4,size,size)
intensityA=torch.Tensor(size,size)
intensityB=torch.Tensor(size,size)



----------------------------------------------
--absolute module
--------------------------------------------

absnet=nn.Sequential()
absnet:add(nn.Abs())
---------------------------------------------


---------------------
-- edgenet module
---------------------

edgenet = nn.Sequential()
m = nn.SpatialConvolution(1,1,3,3,1,1)
m.weight[1][1][1][1] = 0
m.weight[1][1][2][1] = -1
m.weight[1][1][3][1] = 0
m.weight[1][1][1][2] = -1
m.weight[1][1][2][2] = 4
m.weight[1][1][3][2] = -1
m.weight[1][1][1][3] = 0
m.weight[1][1][2][3] = -1
m.weight[1][1][3][3] = 0
m.bias:fill(0)
n = nn.SpatialPadding(1,1,1,1)
o = nn.Abs()
edgenet:add(m)
edgenet:add(n)
edgenet:add(o)

--------------------------
-- edgestrip module
--------------------------

edgestrip = nn.Sequential()
m = nn.SpatialConvolution(1,1,3,3,1,1)
m.weight[1][1][1][1] = 0
m.weight[1][1][2][1] = 0
m.weight[1][1][3][1] = 0
m.weight[1][1][1][2] = 0
m.weight[1][1][2][2] = 1
m.weight[1][1][3][2] = 0
m.weight[1][1][1][3] = 0
m.weight[1][1][2][3] = 0
m.weight[1][1][3][3] = 0
m.bias:fill(0)
edgestrip:add(m)
------------------------------------------------------

-------------------------------------------------
-- These two functions below are borrowed from old attention code
--------------------------------------------------
-- function that receives a 2D tensor "input" 
-- the function returns the location - (r,c) of the 
-- maximum value in the tensor
--------------------------------------------------

function find_max_index(input)

   local max_col = torch.max(input,1)
   local max_val, idx_col = torch.max(max_col, 2)

   local max_row = torch.max(input,2)
   local max_val, idx_row = torch.max(max_row,1)

   return idx_row[1][1], idx_col[1][1]
end

----------------------------------
--alternative method for finding max: divide image into a grid
--find global maxima within subdivisions
--nPoints number of grids
-----------------------------------
function return_section(input)
   local x_range = math.floor((input:size(1))/pointSize)
   local y_range = math.floor((input:size(2))/pointSize)

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

----------------------

-- Matrix to use in sobel operation (Edge detection)
kernel=torch.Tensor(3,3)
n=kernel:storage()
n[1]=0 n[2]=-1 n[3]=0
n[4]=-1 n[5]=4 n[6]=-1
n[7]=0 n[8]=-1 n[9]=0




--copy first image (Note this operation is done only one time so not in the loop)

intensity_dev_a=neuFlow:copyFromHost(intensity)
intensity_dev_b=neuFlow:copyFromHost(intensity)
intensity_dev_c=neuFlow:allocHeap(intensity)

output_dev=neuFlow:allocHeap(output)
intermed_dev=neuFlow:allocHeap(intermediate)

kernel_dev=neuFlow:allocDataPacked(kernel)



-- global loop
neuFlow:beginLoop('main') do

   -- copy input image to dev
   
	neuFlow:copy(intensity_dev_a, intensity_dev_b)
	neuFlow:copyFromHost(intensity, intensity_dev_a)
	
	
	inputs_dev=neuFlow:copyFromHost(inputs)
	intensity_dev=neuFlow:copyFromHost(intensitydiv)
   
   
    -- get RG map
	neuFlow.core:subtract(inputs_dev[1], inputs_dev[2], intermed_dev[1])
	neuFlow.core:divide(intermed_dev[1], intensity_dev[1], output_dev[1])
   
   	--get BY map
	neuFlow.core:add(inputs_dev[1], inputs_dev[2], intermed_dev[2])
	neuFlow.core:multiplyScalar(intermed_dev[2], 0.5, intermed_dev[1])
	neuFlow.core:subtract(inputs_dev[3], intermed_dev[1], intermed_dev[2])
	neuFlow.core:divide(intermed_dev[2], intensity_dev[1], output_dev[2])
   
   	--get Edge map
	neuFlow.core:convolBank({intensity_dev_a[1]}, {kernel_dev[1]}, {intermed_dev[3]})
	output_dev[3]=neuFlow:compile(absnet, {intermed_dev[3]})
   
   	--get Temporal Difference map
	neuFlow.core:subtract(intensity_dev_a[1], intensity_dev_b[1], intermed_dev[1])
  	TD_dev=neuFlow:compile(absnet, {intermed_dev[1]})
   
   
   
   -- copy filtered images to host
	output2=neuFlow:copyToHost(output_dev[2])
	output3=neuFlow:copyToHost(output_dev[1])
	output4=neuFlow:copyToHost(output_dev[3])
	output5=neuFlow:copyToHost(TD_dev)
 
 
	-- multiplication and addition operations to find the saliency map
	neuFlow.core:multiplyScalar(TD_dev[1], 0.4,TD_dev[1])
	neuFlow.core:multiplyScalar(output_dev[3][1], 0.3,output_dev[3][1])
	neuFlow.core:multiplyScalar(output_dev[2], 0.1, output_dev[2])
	neuFlow.core:multiplyScalar(output_dev[1], 0.1, output_dev[1])
	neuFlow.core:multiplyScalar(intensity_dev_a[1], 0.1, intensity_dev_c[1])
	
	neuFlow.core:add(TD_dev[1], output_dev[1], intermed_dev[1])
	neuFlow.core:add(intermed_dev[1],  output_dev[2], intermed_dev[2])
	neuFlow.core:add(intermed_dev[2],   intensity_dev_c[1], intermed_dev[3])
	neuFlow.core:add(intermed_dev[3],  output_dev[3][1], intermed_dev[4])
	
	-- copy salience to host
	sal_dev=neuFlow:compile(edgestrip, {intermed_dev[4]})
	salience=neuFlow:copyToHost(sal_dev[1])
	
	
end neuFlow:endLoop('main')



-- LOAD: load the bytecode on the device, and execute it
neuFlow:loadBytecode()


----------------------------------------------------------------------
-- EXEC: this part executes the host code, and interacts with the dev
--



camera=image.Camera{}


function process()
	neuFlow.profiler: start('whole-loop', 'fps')
	
	
	-- get frame from camera and obtain intensity image by rgb function
   camFrame = camera:forward()
   camFrameYa = image.rgb2y(camFrame)
   
   image.scale(camFrameYa, intensitydiv, 'bilinear')
   intensitydiv=intensitydiv+1
   camFrameY=camFrameYa:select(1,1)
   

   image.scale(camFrameY, intensity, 'bilinear')
   image.scale(camFrame, inputs, 'bilinear')

	-- Remember copyHost() inserted in device parts need to 
	--be matched by a copyDev in this execution section

	-- In the device part copying first image only occurs one time
	-- so to match the copyHost()-copyDev()  we should send this data only once
	-- This part provides that matching, if target is neuflow for the first time we send the intensity maps to device  
   
  	if (target=='neuflow') and (counting==0) then
   		neuFlow:copyToDev(intensity)
   		neuFlow:copyToDev(intensity)
   		counting=1
   	end
   
 	if (target=='neuflow') then	
		neuFlow:copyToDev(intensity)
		neuFlow:copyToDev(inputs)
		neuFlow:copyToDev(intensitydiv)

	-- Copy results from the device to display
		neuFlow:copyFromDev(output2)
		neuFlow:copyFromDev(output3)
		neuFlow:copyFromDev(output4)
		neuFlow:copyFromDev(output5)
	
		neuFlow:copyFromDev(salience)
		
	else

	one_map = torch.Tensor(1,size,size)
	two_map = torch.Tensor(1,size,size)
	three_map = torch.Tensor(1,size,size)
	four_map = torch.Tensor(1,size,size)
	I_map = torch.Tensor(1,size,size)
		profiler_cpu = neuFlow.profiler:start('compute-cpu')
		neuFlow.profiler:setColor('compute-cpu', 'blue')
   
	intensityA:copy(intensityB)
	intensityB:copy(intensity)

		--get Temporal Difference map
		output5 = (intensityB - intensityA):abs()
		output5:resize(1,output5:size(1), output5:size(2))
		one_map:copy(output5)
  
		-- get RG map
		output3 = (inputs:select(1,1) - inputs:select(1,2)):cdiv(intensitydiv)
		output3:resize(1, output3:size(1), output3:size(2))   
		two_map:copy(output3)
		
		-- get BY map
		mid = ((inputs:select(1,1))+(inputs:select(1,2))):mul(0.5)
		output2 = ((inputs:select(1,3))-(mid)):cdiv(intensitydiv)
		output2:resize(1,output2:size(1), output2:size(2))
		three_map:copy(output2) 

		-- get Edge map
		intensitymap = torch.Tensor(1,size,size)
		intensitymap:copy(intensityB)
		output4 = edgenet:forward((intensitymap))
		four_map:copy(output4)

		I_map:copy(intensitydiv-1)   

	-- multiplication and addition operations to find the saliency map
	temp = (one_map:mul(0.4)):add(two_map:mul(0.1)):add(three_map:mul(0.1)):add(I_map:mul(0.1)):add(four_map:mul(0.3))
	salience = edgestrip:forward(temp)   

   neuFlow.profiler:lap('compute-cpu')
   end


	
	sal=salience[1]		
	neuFlow.profiler:start('scale-salience-down')
	saliencetemp = return_section(sal)
	neuFlow.profiler:lap('scale-salience-down')
			
   	win:gbegin()
   	win:showpage()
   		
  	neuFlow.profiler:start('display')
  	
	-- display original input image
	image.display{image=inputs, win=win,
   	                zoom=1,  x=0, y=30, legend='Input'}
			
		   win:moveto(0, 20)
	   	   win:setfont(qt.QFont{size=14})
   		   win:show(string.format('Input'))    
		   
	-- display BY 
	image.display{image=output2, win=win,
   	                zoom=0.4,  x=500, y=30, legend='BY'}
   	       win:moveto(500, 20)
   		   win:show(string.format('BY'))  
  
	-- display RG 			
	image.display{image=output3, win=win,
   	                zoom=0.4,  x=530+output3:size(2)*0.4, y=30, legend='RG'}
   	       win:moveto(530+output3:size(2)*0.4, 20)
   		   win:show(string.format('RG')) 

	-- display sobel image 			
	image.display{image=output4, win=win,
   	                zoom=0.4,  x=500, y=60+output3:size(3)*0.4, legend='Edge'}
   	       win:moveto(500, 50+output3:size(3)*0.4)
   		   win:show(string.format('Edge')) 
   	        
	-- display temporal difference image 		
	image.display{image=output5, win=win,
  	                zoom=0.4,  x=530+output3:size(3)*0.4, y=60+output3:size(3)*0.4, legend='Temp Diff'}
  	       win:moveto(530+output3:size(3)*0.4, 50+output3:size(3)*0.4)
   		   win:show(string.format('Temp Diff')) 
	
	-- display Salinecy map 		
	image.display{image=salience, win=win,
  	                zoom=0.4,  x=530, y=90+output3:size(3)*0.8, legend='Salience'}
		   win:moveto(530,80+output3:size(3)*0.8)
   		   win:show(string.format('Salience'))
		
		
		
		-- draw squares 
	    for i=1,nPoints do
	    
	    	local x, y = find_max_index(saliencetemp)
	    	
	    	saliencetemp[x][y]=0	    	
			x = 1+pointSize*(x-1)
	    	y = 1+pointSize*(y-1)	    		    	
	    	 win:rectangle(y, x+30, pointSize, pointSize)
	    	 win:stroke()
	  
	    end
      
	    
	    neuFlow.profiler:lap('display')
	    neuFlow.profiler:lap('whole-loop')
	    
	    neuFlow.profiler:displayAll{win=win, x=0, y=450, zoom=0.6}



	local x = 30
   	local y = 650
   	local zoom = 0.6
   	win:setfont(qt.QFont{serif=false,italic=false,size=24*zoom})   
 	win:setcolor("red")
   	local str
   	str = string.format('compare to <%s> = %f fps', last_name, last_fps)
   
   -- disp line:
   	win:moveto(x,y);
  	win:show(str)
   
   -- if we have both cpu and neuflow timimgs saved
   	if(last_fps ~= -1) then
      	x = 400
      	y = 650
      	win:setfont(qt.QFont{serif=false,italic=false,size=28*zoom})
    	win:setcolor("red")      
      	local speedup = neuFlow.profiler.events['compute-cpu'].reald/neuFlow.profiler.events['on-board-processing'].reald
     	str = string.format('speedup = %f ',
			  speedup)    
   -- disp line:
		win:moveto(x,y);
		win:show(str)
   	end
  
	win:gend()
end   

-- window
-- setup GUI (external UI file)
widget = qtuiloader.load('attention.ui')
win = qt.QtLuaPainter(widget.frame)



-- Create QT events
local timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
qt.connect(timer,'timeout()',			
           function() 
           
            	
             	nPoints = widget.spinBox.value
              	pointSize = widget.spinBox_2.value
             	checkerror = widget.checkBox.checked
             	process()
              	timer:start()
           end )
	   
qt.connect(qt.QtLuaListener(widget.pushButton_2),
           'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
           function (...) 
	      -- only if we are changing the process source,
	      --change the fps
	      if (target ~= 'neuflow') then
			last_fps = 1/neuFlow.profiler.list[1].reald
	      end
          target = 'neuflow'
	      last_name = 'cpu'
	   end)

qt.connect(qt.QtLuaListener(widget.pushButton),
           'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
           function (...) 
	      -- only if we are changing the process source,
	      --change the fps
	      if (target ~= 'cpu') then
		 	last_fps = 1/neuFlow.profiler.list[1].reald
	      end
          target = 'cpu'
	      last_name = 'neuflow'
	   end)



-- Close Process
qt.connect(qt.QtLuaListener(widget),
           'sigShow(bool)',
		function(b) 
		if b then timer:start() end 
		end )



timer:start()
widget.windowTitle = "Temporal Difference Imaging"
widget:show()


