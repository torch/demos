---------------------------------------------------------------


-- Segmentation
-- The image is grabbed from a webcam and the segmentation run on Neuflow. 
--
-- Author: Aysegul Dundar
-- Date: 16/03/2012


require 'neuflow'
require 'xlua' 
require 'os'
require 'torch'
require 'qt'


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

----------------------------------------------------------------------
-- INIT: initialize the neuFlow context
-- a mem manager, the dataflow core, and the compiler


neuFlow = neuflow.init()

----------------------------------------------------------------------
-- ELABORATION: describe the algorithm to be run on neuFlow, and 
-- how it should interact with the host (data exchange)
-- note: any copy**Host() inserted here needs to be matched by
-- a copy**Dev() in the EXEC section.



-- image sizes

size = 380
pd = torch.Tensor(1,380,380):fill(64)
--pd[1][1]=2
--pd[1]=2
input = torch.Tensor(3,size,size)
intermed_dev1=neuFlow:allocHeap(input)
intermed_dev2=neuFlow:allocHeap(input)
pd_dev=neuFlow:allocDataPacked(pd)


-- global loop
neuFlow:beginLoop('main') do

   	-- copy input image to dev  	
	input_dev=neuFlow:copyFromHost(input)
 
	-- division and multiplication for segmentation
	neuFlow.core:divide(input_dev[1], pd_dev[1], intermed_dev1[1])
	neuFlow.core:divide(input_dev[2], pd_dev[1], intermed_dev1[2])
	neuFlow.core:divide(input_dev[3], pd_dev[1], intermed_dev1[3])
	
	neuFlow.core:multiplyScalar(intermed_dev1[1], 64, intermed_dev2[1])
	neuFlow.core:multiplyScalar(intermed_dev1[2], 64, intermed_dev2[2])
	neuFlow.core:multiplyScalar(intermed_dev1[3], 64, intermed_dev2[3])
	
	output=neuFlow:copyToHost(intermed_dev2)
	--output=neuFlow:copyToHost(intermed_dev2[1])
	
	
end neuFlow:endLoop('main')



-- LOAD: load the bytecode on the device, and execute it
neuFlow:loadBytecode()


----------------------------------------------------------------------
-- EXEC: this part executes the host code, and interacts with the dev


camera = image.Camera(opt.camidx)


function process()
	neuFlow.profiler: start('whole-loop', 'fps')
	

   	camFrame = camera:forward()
	image.scale(camFrame, input, 'simple')
 
	neuFlow:copyToDev(input)		
	neuFlow:copyFromDev(output)
		

   	win:gbegin()
   	win:showpage()
   		
  	neuFlow.profiler:start('display')
  	
	-- display original input image
	image.display{image=input, win=win,
   	                zoom=1,  x=0, y=30, legend='Input'}
			
		   win:moveto(0, 20)
	   	   win:setfont(qt.QFont{size=14})
   		   win:show(string.format('Input'))    
		   
	-- display segmented 
	image.display{image=output, win=win,
   	                zoom=1,  x=30+camFrame:size(2), y=30, legend='Segmented'}
   	       win:moveto(30+camFrame:size(2), 20)
   		   win:show(string.format('Segmented'))  
  
	    
	    neuFlow.profiler:lap('display')
	    neuFlow.profiler:lap('whole-loop')
	    

  
	win:gend()
end   

-- window
-- setup GUI (external UI file)
--widget = qtuiloader.load('segmentation.ui')
--win = qt.QtLuaPainter(widget.frame)

win = qtwidget.newwindow(900,540,'Segmentation')

-- Create QT events
local timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
qt.connect(timer,'timeout()',			
           function() 
			process() 
			timer:start()          	
           end )
	   



--widget.windowTitle = "Segmentation"
--widget:show()

timer:start()

