------------------------------------------------------------
-- Road-Net demo by e-Lab
--
-- Wed Nov 13 10:28:14 EST 2013
-- 
-- E. Culurciello, Alfredo Canziani, Artem Kuharenko
-- original code and net training by Clement Farabet
--
------------------------------------------------------------

require 'pl'
require 'qt'
require 'qtwidget'
require 'ffmpeg'
require 'imgraph' -- to colorize outputs
require 'segmtools' -- to display name of classes
require 'nnx'
require 'image'
require 'camera'

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
  -t,   --threads       (default 3)          number of threads
  -v,   --video         (default '')         video (or image) file to process
  -n,   --network       (default 'multinet-float.net') path to trained network
        --networktype   (default 'cnn')      type of network ('cnn' or 'unsup')
  -s,   --save          (default '')         path to save output video
  -l,   --useffmpeglib  (default false)      help=use ffmpeglib module to read frames directly from video
  -k,   --seek          (default 0)          seek number of seconds
  -f,   --fps           (default 10)         number of frames per second
        --seconds       (default 10)         length to process (in seconds)
  -w,   --width         (default 320)        resize video, width 
  -h,   --height        (default 200)        resize video, height
  -z,   --zoom          (default 1)          display zoom
        --downsampling  (default 2)          downsample input frame for processing
  -c,   --camidx        (default 0)          if source=camera, specify the camera index: /dev/videoIDX
]]
opt.downsampling = tonumber(opt.downsampling)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

-- ANCIENT FUNCTION FROM OLD SYS to make ffmpeg work:
-- also had to install ancient lua--sys for all this to work: 
-- https://github.com/clementfarabet/lua---sys/tree/a9ec80446d0482f9b5e48d1e6db67b8c4f9cd126
--------------------------------------------------------------------------------
-- file iterator, in given path
--------------------------------------------------------------------------------
function sys.files(path)
   local d = sys.dir(path)
   local n = 0
   return function()
             n = n + 1
             if (d and n <= #d) then
                return d[n]
             else
                return nil
             end
          end
end


---------------------------------------------------------------------------------
--Define cnn functions
---------------------------------------------------------------------------------
function init_cnn_net()

	-- load pre-trained network and other data from disk
	print(opt.network)
	local netin = torch.load(opt.network)

	-- replace classifier (2nd module) by SpatialClassifier
	local network = netin.modules[1]
	local classifier1 = netin.modules[2]
	local classifier = nn.SpatialClassifier(classifier1)

	-- replace SpatialCconvolutionMM:
	local m1 = network.modules[1]:clone()
	network.modules[1] = nn.SpatialConvolution(3,16,7,7)
	network.modules[1].weight = m1.weight:reshape(16,3,7,7)
	network.modules[1].bias = m1.bias
	m1 = network.modules[4]:clone()
	network.modules[4] = nn.SpatialConvolution(16,64,7,7)
	network.modules[4].weight = m1.weight:reshape(64,16,5,5)
	network.modules[4].bias = m1.bias
	m1 = network.modules[7]:clone()
	network.modules[7] = nn.SpatialConvolution(64,256,5,5)
	network.modules[7].weight = m1.weight:reshape(256,64,5,5)
	network.modules[7].bias = m1.bias
	
	local classes = {'unknown', 'building', 'bus', 'car', 'grass', 'person', 'road', 'sign', 'sky', 'tree'}
	local colours = {[ 1] = {0.0, 0.0, 0.0},
		        [ 2] = {0.7, 0.7, 0.3}, -- building
		        [ 3] = {0.4, 0.7, 0.8}, -- bus
		        [ 4] = {0.4, 0.4, 0.8}, -- car
		        [ 5] = {0.0, 0.9, 0.0}, -- grass
		        [ 6] = {1.0, 0.0, 0.3}, -- person
		        [ 7] = {0.3, 0.3, 0.3}, -- road
		        [ 8] = {1.0, 0.1, 0.1}, -- sign
		        [ 9] = {0.0, 0.7, 0.9}, -- sky
		        [10] = {0.2, 0.8, 0.1}} -- tree

	local alg_data = {}
	alg_data.network = network
	alg_data.classifier = classifier

	return alg_data, classes, colours

end

function cnn_preprocess(alg_data, frame)
	return frame
end

function get_cnn_features(alg_data, frame)
	return alg_data.network:forward(frame)
end

function get_cnn_distributions()

   local distributions = alg_data.classifier:forward(features)
   distributions = nn.SpatialClassifier(nn.SoftMax()):forward(distributions)
	return(distributions)

end
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
--Define unsupervised net functions
---------------------------------------------------------------------------------
function init_unsup_net()

   local data = {}
   print('loading unsup model')
   data.preproc = torch.load('preproc.t7')
   data.u1net = torch.load('u1net.net')
   data.classifier = torch.load('unsup-cl-30.net')

   local classes = {'building', 'car', 'grass', 'person', 'road', 'sign', 'sky', 'tree'}

   local colours = {[ 1] = {0.7, 0.7, 0.3}, -- building
				        [ 2] = {0.4, 0.4, 0.8}, -- car
				        [ 3] = {0.0, 0.9, 0.0}, -- grass
				        [ 4] = {1.0, 0.0, 0.3}, -- person
							--[4]={0,0,0},
				        [ 5] = {0.3, 0.3, 0.3}, -- road
				        [ 6] = {1.0, 0.1, 0.1}, -- sign
				        [ 7] = {0.0, 0.7, 0.9}, -- sky
				        [ 8] = {0.2, 0.8, 0.1}} -- tree

   return data, classes, colours

end

function unsup_preprocess(alg_data, frame)

   for i = 1,3 do
      frame[{ {i},{},{} }]:add(-alg_data.preproc.mean[i])
      frame[{ {i},{},{} }]:div(alg_data.preproc.std[i])
   end

   return frame

end

function get_unsup_features(alg_data, frame)

	features = alg_data.u1net:forward(frame)
   return features

end


function get_unsup_distributions(alg_data, features)

	d = 12

	stride = 12
	nfy = math.floor(features:size(2) / stride) 
	nfx = math.floor(features:size(3) / stride) 
   
	local distributions = torch.Tensor(8, nfy, nfx)
	local f_list = torch.Tensor(nfx * nfy, features:size(1), d, d)

	local i = 0
	--reoder features
   for y = 1, nfy do
      for x = 1, nfx do
			
			i = i + 1
			x1 = (x - 1) * stride + 1
			x2 = x1 + d - 1
			y1 = (y - 1) * stride + 1
			y2 = y1 + d - 1
			f_list[i] = features[{{},{y1, y2}, {x1, x2}}] -- if stride >= d then we could just copy references

      end
   end

	--compute distributions
	local d_list = alg_data.classifier:forward(f_list) 

	--reoder distributions
	i = 0
   for y = 1, nfy do
      for x = 1, nfx do

			i = i + 1
			distributions[{{}, {y}, {x}}] = d_list[i]

		end
	end

   return distributions

end
---------------------------------------------------------------------------------

--choose algorithm: 
if opt.networktype == 'unsup' then

	--unsup
	init_net = init_unsup_net
	preprocess_frame = unsup_preprocess
	get_features = get_unsup_features
	get_distributions = get_unsup_distributions

else 

	--cnn
	init_net = init_cnn_net
	preprocess_frame = cnn_preprocess
	get_features = get_cnn_features
	get_distributions = get_cnn_distributions

end
---------------------------------------------------------------------------------

alg_data, classes, colours = init_net()

-- generating the <colourmap> out of the <colours> table
colormap = imgraph.colormap(colours)

-- load video
if opt.video ~= '' then
   video = ffmpeg.Video{path=opt.video,
   width=opt.width, height=opt.height,
   fps=opt.fps, length=opt.seconds, seek=opt.seek,
   encoding='jpg',
   delete=false}
else
   camera = image.Camera{}
end

-- setup GUI (external UI file)
if not win or not widget then
   win = qtwidget.newwindow(opt.width*opt.zoom, opt.height*opt.zoom*2 + 80, -- 20 for class display
   'E-Lab RoadNet demo')
   font = qt.QFont{serif=false, italic=false, size=12}
   win:setfont(font)
end

-- prepare classes caption at bottom of window:
-- ffclasses = torch.Tensor(3, 20, opt.width*opt.zoom)
-- for i = 1,#classes do
--   local dx = 51
--   local x = 1+(i-1)*dx
--   if (x > (#ffclasses)[3]) then x = (#ffclasses)[3] end
--   print(x)
--   ffclasses[{{1},{1,20},{x,x+dx}}]:fill(colours[i][1])
--   ffclasses[{{2},{1,20},{x,x+dx}}]:fill(colours[i][2])
--   ffclasses[{{3},{1,20},{x,x+dx}}]:fill(colours[i][3])
-- end

local hwt = 0

-- process and time in SW on CPU:
cput = 0
for i = 1, 10 do
	sys.tic() --test on HW
	features = get_features(alg_data, img_temp)
	cput = cput + sys.toc()
end
cput = cput / 10
print('CPU frame precessing time [ms]: ', cput*1000)


-- process function
function process()
   -- grab frame
   if opt.video ~= '' then
      fframe = video:forward()
   else
      fframe = camera:forward()
   end

   local width = opt.width
   local height = opt.height

   cframe = fframe
   if opt.downsampling > 1 then
      width  = width/opt.downsampling
      height = height/opt.downsampling
      frame = image.scale(cframe, width, height)
   else
      frame = cframe:clone()
   end

	frame = preprocess_frame(alg_data, frame)

   -- process frame with network:  
	sys.tic()
    features = get_features(alg_data, frame)
	print('features ' .. sys.toc())   
  
   -- (a) compute class distributions
	sys.tic()
	distributions = get_distributions(alg_data, features)
	print('distributions ' .. sys.toc())   

   -- (b) upsample the distributions
   distributions = image.scale(distributions, frame:size(3), frame:size(2), 'simple')

   -- (d) winner take all
   _,winners = torch.max(distributions,1)
   winner = winners[1]
end


-- display function
function display()
   -- colorize classes
   colored, colormap = imgraph.colorize(winner, colormap)

   -- display raw input
   image.display{image=fframe, win=win, zoom=opt.zoom, min=0, max=1}

   -- map just the processed part back into the whole image
   if opt.downsampling > 1 then
      colored = image.scale(colored,fframe:size(3),fframe:size(2))
   end
   colored:add(fframe)
   
   -- overlay segmentation on input frame
   image.display{image=colored, win=win, y=fframe:size(2)*opt.zoom, zoom=opt.zoom, min=0, max=colored:max()}

   -- print classes:
   -- this if you want to print predefine color labels ffclasses:
   --image.display{image=ffclasses, win=win, y=2*fframe:size(2)*opt.zoom, zoom=opt.zoom, min=0, max=1}
   for i = 1,#classes do
      local dx = 52
      local x = (i-1)*dx
      win:rectangle(x, opt.height*opt.zoom*2, dx, 20)--opt.height*opt.zoom*2+20)
      win:setcolor(colours[i][1],colours[i][2],colours[i][3])
      win:fill()
      win:setcolor('black')--colours[i])
      win:moveto(x+5, opt.height*opt.zoom*2 + 15)
      win:show(classes[i])
   end

   -- display profile data:
   local speedup = cput/hwt
   str1 = string.format('CPU compute time [ms]: %f',  cput*1000)
   str2 = string.format('HW config+compute time [ms]: %f', hwt*1000)
   str3 = string.format('speedup = %f ',  speedup)
   win:setfont(qt.QFont{serif=false,italic=false,size=12})
   -- disp line:
   win:moveto(10, opt.height*opt.zoom*2 + 35);
   win:show(str1)
   win:moveto(10, opt.height*opt.zoom*2 + 55);
   win:show(str2)
   win:moveto(10, opt.height*opt.zoom*2 + 75);
   win:show(str3)
   
   -- save ?
   if opt.save ~= '' then
      local t = win:image():toTensor(3)
      local fname = opt.save .. string.format('/frame_%05d.jpg',times)
      sys.execute(string.format('mkdir -p %s',opt.save))
      print('saving:'..fname)
      image.save(fname,t)
   end
end

-- setup gui
while win:valid() do
   process()
   win:gbegin()
   win:showpage()
   display()
   win:gend()
   collectgarbage()
end

