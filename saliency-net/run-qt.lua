#!/usr/bin/env qlua
------------------------------------------------------------
-- a simple convnet saliency demo
-- takes output of convnet 1st, 2nd layers and displays it
-- uses Qt graphics
-- uses roadnet network for better "objectness"
-- 
-- TODO: ADD motion channel!
--
-- E. Culurciello, 2013
-- 
------------------------------------------------------------

require 'pl'
require 'qt'
require 'qtwidget'
require 'nnx'
require 'image'
require 'ffmpeg'
require 'camera'

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
  -x,   --runnnx        (default true)       run on hardware nn_X 
  -t,   --threads       (default 8)          number of threads
  -v,   --video         (default '')         video (or image) file to process
  -n,   --network       (default '../../Road-net/Demo/multinet-float.net') path to trained network
  -s,   --save          (default '')         path to save output video
  -w,   --width         (default 320)        resize video, width 
  -h,   --height        (default 200)        resize video, height
  -z,   --zoom          (default 1)          display zoom
        --downsampling  (default 2)          downsample input frame for processing
  -c,   --camidx        (default 0)          if source=camera, specify the camera index: /dev/videoIDX
]]
opt.downsampling = tonumber(opt.downsampling)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)

-- setup camera
local iW = opt.width
local iH = opt.height
-- load video
if opt.video ~= '' then
      video = {}
      video.forward = function()
                        return i
                      end
      video = ffmpeg.Video{path=opt.video, 
                        width=1800, height=1000,
                        fps=opt.fps, length=opt.seconds, seek=opt.seek, 
                        encoding='jpg',
                        delete=false}
else 
  camera = image.Camera{}
end
local img1 = torch.Tensor(3,iH,iW)
local img2 = torch.Tensor(3,iH,iW)
local frame = torch.Tensor(iH,iW) -- filter is 5x5 so lose 2 pix on each sides

-- load network and clone 1st layer
local netin = torch.load(opt.network):float()

-- replace classifier (2nd module) by SpatialClassifier
network = netin.modules[1]
--local classifier1 = netin.modules[2]
--classifier = nn.SpatialClassifier(classifier1)

--network.modules[5] = nil
--network.modules[6] = nil
network.modules[7] = nil
network.modules[8] = nil
network.modules[9] = nil

network:add(nn.Sum(1))

-- replace SpatialCconvolutionMM:
m1 = network.modules[1]:clone()
network.modules[1] = nn.SpatialConvolution(3,16,7,7)
network.modules[1].weight = m1.weight:reshape(16,3,7,7)
network.modules[1].bias = m1.bias
m1 = network.modules[4]:clone()
network.modules[4] = nn.SpatialConvolution(16,64,7,7)
network.modules[4].weight = m1.weight:reshape(64,16,5,5)
network.modules[4].bias = m1.bias
-- m1 = network.modules[7]:clone()
-- network.modules[7] = nn.SpatialConvolution(64,256,5,5)
-- network.modules[7].weight = m1.weight:reshape(256,64,5,5)
-- network.modules[7].bias = m1.bias


-- process function
function process()
   img2=img2:copy(img1)
   --get frame:
   if opt.video ~= '' then
      img1 = image.scale(image.crop(video:forward(), 1200, 500, 1800, 1000),iW,iH)
   else
      img1 = image.scale(camera:forward(),iW,iH)
   end
   -- process:
   nout = network:forward(img1) -- pass RGB
   out = image.scale(nout, iW, iH)
   frame:mul(0):add(out):mul(0.5) -- :add(-torch.min(frame)):div(torch.max(frame)):mul(0.2) -- reset, then normalize
   --colorop =  (img1[2]-img1[1]) + (img1[2]-img1[3]) -- color opponency
   --colorop:mul(0.5)--:add(-torch.min(colorop)):div(torch.max(colorop)):mul(0.5) -- normalize
   --tmpdiff = (img1[2]-img2[2]) -- temp diff
   --tmpdiff:add(-torch.min(tmpdiff)):div(torch.max(tmpdiff)):mul(0.6) -- normalize
   --frame:add(colorop)--:add(tmpdiff):add(colorop) -- add temp diff, color opp
end   

-- setup GUI (external UI file)
if not win or not widget then
   win = qtwidget.newwindow(opt.width*opt.zoom, opt.height*opt.zoom*2 + 80, -- 20 for class display
   'E-Lab Attention Net demo')
   font = qt.QFont{serif=false, italic=false, size=12}
   win:setfont(font)
end

-- display function
function display()
   image.display{image=frame, win=win, zoom=opt.zoom}
end

-- display loop:
while true do
      process()
      display()
end

