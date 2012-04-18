-- A simple test code for gabor filters
-- author: Prashant Lalwani

--Description:

-- Real part of a gabor filter: 
--	g(x,y,lambda,theta,shi,sigma,gamma) = exp{-(xPrime^2 + gamma^2 * yPrima^2)/(2 * sigma^2)} * cosine(2*pi*xPrime/lambda + shi)
-- 	xPrime =  xcos(theta) + ysin(theta)
-- 	yPrime = -xsin(theta) + ycos(theta)

-- 	lambda represents the wavelength of the filter, values above 2 and less than 1/5 of image size are valid
		--	Its value is specified in pixels

--	theta represents orientation of the normal to the paralle stripes of a gabor function
		--	Its value is specified in degrees so the valid range becomes 0-360

--	shi is the phase offset, valid values are real numbers between -180 and 180
		--	The values 0 and 180 correspond to center-symmetric center-on and center-off functions respectively
		-- 	It can be used to change the intensity of the background with respect to the filter

--	sigma is the gaussian envelope, sigma = 0.56*lambda should work in our case which gives a bandwidth of 1
		--	sigma = 0.56*lambda = 4.48 for a bandwidth of 1 so 1/(2*sigma^2) = 0.02 as used in equation 3

--	gamma is the spatial aspect ratio and defines the ellipticity of the filter i.e. gamma = 1 will give a circle


-- include packages
require 'xlua' 
require 'os'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
xrequire('nnx',true)
xrequire('camera',true)

-- global declarations
pi = 3.14
gamma = 0.5
shi = 0

-- setup GUI (external UI file)
widget = qtuiloader.load('frame.ui')
win1 = qt.QtLuaPainter(widget.frame)

-- initializing the camera
camera = image.Camera{}

-- Gabor filter algorithm to compute the Tensor
function GaborLayer(Sx,Sy,lambda,theta)
	sigma = 0.56*lambda
	Gabor = torch.Tensor(Sx,Sy)
	for x = 1,Sx do
		for y = 1,Sy do
	xPrime =  (x-Sx/2-1)*math.cos(theta) + (y-Sy/2-1)*math.sin(theta)	--equation 1
	yPrime = -(x-Sx/2-1)*math.sin(theta)  + (y-Sy/2-1)*math.cos(theta)	--equation 2
	 Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime^2)+(yPrime^2 * gamma^2 )))*math.cos(2*pi*xPrime/lambda  + shi)	-- equation 3 	
	end
		end	
	return(Gabor)
end

-- Display function
function display()
	cam = torch.Tensor(100,100)
	cam = camera:forward()	-- takes the image from camera
	image.display{image = cam, win = win1, zoom = 0.5}	-- original image

-- First set of images 
	G1 = GaborLayer(9,9,2,0)	-- function call
	realgabor1 = image.convolve(cam,G1,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor1, win = win1, zoom = 0.25, x= 320}

	G2 = GaborLayer(9,9,2,45)	-- function call
	realgabor2 = image.convolve(cam,G2,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor2, win = win1, zoom = 0.25, x= 480}

	G3 = GaborLayer(9,9,2,80)	-- function call
	realgabor3 = image.convolve(cam,G3,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor3, win = win1, zoom = 0.25, x= 320, y =120}

	G4 = GaborLayer(9,9,2,150)	-- function call
	realgabor4 = image.convolve(cam,G4,'valid')	-- convolution of the image with the filter	
	image.display{image = realgabor4, win = win1, zoom = 0.25, x= 480, y = 120}

-- Second set of images
	G21 = GaborLayer(9,9,3,0)	-- function call
	realgabor21 = image.convolve(cam,G21,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor21, win = win1, zoom = 0.25, x= 320, y = 240}

	G22 = GaborLayer(9,9,3,45)	-- function call
	realgabor22 = image.convolve(cam,G22,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor22, win = win1, zoom = 0.25, x= 480, y = 240}

	G23 = GaborLayer(9,9,3,80)	-- function call
	realgabor23 = image.convolve(cam,G23,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor23, win = win1, zoom = 0.25, x= 320, y =360}

	G24 = GaborLayer(9,9,3,150)	-- function call
	realgabor24 = image.convolve(cam,G24,'valid')	-- convolution of the image with the filter	
	image.display{image = realgabor24, win = win1, zoom = 0.25, x= 480, y = 360}


-- Third set of images
	G31 = GaborLayer(9,9,4,0)	-- function call
	realgabor31 = image.convolve(cam,G31,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor31, win = win1, zoom = 0.25, x= 320, y = 480}

	G32 = GaborLayer(9,9,4,45)	-- function call
	realgabor32 = image.convolve(cam,G32,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor32, win = win1, zoom = 0.25, x= 480, y = 480}

	G33 = GaborLayer(9,9,4,80)	-- function call
	realgabor33 = image.convolve(cam,G33,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor33, win = win1, zoom = 0.25, x= 320, y =600}

	G34 = GaborLayer(9,9,4,150)	-- function call
	realgabor34 = image.convolve(cam,G34,'valid')	-- convolution of the image with the filter	
	image.display{image = realgabor34, win = win1, zoom = 0.25, x= 480, y = 600}


-- Fourth set of images
	G41 = GaborLayer(9,9,8,0)	-- function call
	realgabor41 = image.convolve(cam,G41,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor41, win = win1, zoom = 0.25, x= 320, y = 720}

	G42 = GaborLayer(9,9,8,45)	-- function call
	realgabor42 = image.convolve(cam,G42,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor42, win = win1, zoom = 0.25, x= 480, y = 720}

	G43 = GaborLayer(9,9,8,80)	-- function call
	realgabor43 = image.convolve(cam,G43,'valid')	-- convolution of the image with the filter
	image.display{image = realgabor43, win = win1, zoom = 0.25, x= 320, y =840}

	G44 = GaborLayer(9,9,8,150)	-- function call
	realgabor24 = image.convolve(cam,G44,'valid')	-- convolution of the image with the filter	
	image.display{image = realgabor24, win = win1, zoom = 0.25, x= 480, y = 840}

end

-- qt timer
timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,'timeout()', function() display() timer:start() end)

-- loads the widget
widget.windowTitle = 'A Test widget'
widget:show()
timer:start()