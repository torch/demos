#!/usr/bin/env qlua
------------------------------------------------------------
-- a simple convnet saliency demo
-- takes output of convnet 1st layer and displays it

require 'camera'
require 'nnx'
require 'ffmpeg'
require 'luasdl'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, you can specify the camera index: /dev/videoIDX', 
          default=0}
op:option{'-n', '--network', action='store', dest='network', 
          help='path to existing [trained] network',
          default='../face-detector/face.net'}
          --default='/Users/eugenioculurciello/Code/torch/neuflow-demos/scene-parsing/stanford.net'}
op:option{'-v', '--video', action='store', dest='video',
          help='video file to process'}
op:option{'-k', '--seek', action='store', dest='seek',
          help='seek number of seconds', default=0}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=10}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=10}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=600}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=600}
op:option{'-z', '--zoom', action='store', dest='zoom',
          help='display zoom', default=1}
opt,args = op:parse()

torch.setdefaulttensortype('torch.FloatTensor')

-- profiler
p = xlua.Profiler()

-- init display SDL:
luasdl.init(1200, 600)

-- setup camera
S_x = opt.width
S_y = opt.height
-- load video
if opt.video then
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
img1=torch.Tensor(3,S_y,S_x)
img2=torch.Tensor(3,S_y,S_x)
frame=torch.Tensor(S_y,S_x) -- filter is 5x5 so lose 2 pix on each sides

-- load network and clone 1st layer
network = torch.load(opt.network):float()
net2 = nn.Sequential()
net2:add(network.modules[1])
net2:add(network.modules[2])
net2:add(nn.Sum(1))

-- process function
function process()
   img2=img2:copy(img1)
   --get frame:
   if opt.video then
      img1 = image.scale(image.crop(video:forward(), 1200, 500, 1800, 1000),S_x,S_y)
   else
      img1 = image.scale(camera:forward(),S_x,S_y)
   end
   -- process:
   nout = net2:forward(img1[2]) -- pass RGB
   out = image.scale(nout, S_x, S_y)
   frame:mul(0):add(out):mul(0.5) -- :add(-torch.min(frame)):div(torch.max(frame)):mul(0.2) -- reset, then normalize
   --colorop =  (img1[2]-img1[1]) + (img1[2]-img1[3]) -- color opponency
   --colorop:mul(0.5)--:add(-torch.min(colorop)):div(torch.max(colorop)):mul(0.5) -- normalize
   --tmpdiff = (img1[2]-img2[2]) -- temp diff
   --tmpdiff:add(-torch.min(tmpdiff)):div(torch.max(tmpdiff)):mul(0.6) -- normalize
   --frame:add(colorop)--:add(tmpdiff):add(colorop) -- add temp diff, color opp
end   

-- display function
function display()
   luasdl.display(torch.cat(img1, frame:reshape(1,S_x,S_y):expand(3,S_x,S_y)))
end

-- display loop:
while true do
      p:start('full loop','fps')
      p:start('prediction','fps')
      process()
      p:lap('prediction')
      p:start('display','fps')
      display()
      p:lap('display')
      p:lap('full loop')
      p:printAll()
end

