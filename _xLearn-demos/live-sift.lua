#!/usr/bin/env qlua
----------------------------------------------------------------------
-- WHAT: computes and finds SIFT keypoints/descriptors
--       the image source must be provided on the command line
--
require 'XLearn'
require 'os'
require 'vlfeat'

-- parse args
op = OptionParser('%prog -s SOURCE [options]')
op:add_option{'-s', '--source', action='store', dest='source',
              help='image source, can be one of: camera | lena | video'}
op:add_option{'-c', '--camera', action='store', dest='camidx',
              help='if source=camera, you can specify the camera index: /dev/videoIDX [default=0]'}
op:add_option{'-p', '--path', action='store', dest='path',
              help='path to video'}
options,args = op:parse_args()

-- setup QT gui
toolBox.useQT()
widget = qtuiloader.load('live-sift.ui')
painter = qt.QtLuaPainter(widget.frame)

-- params
width = 320
height = 180

-- video source
source = nn.ImageSource{type = options.source or 'camera',
                        path = options.path,
                        cam_idx = options.camidx,
                        fps = 20,
                        width = width,
                        height = height}

-- displayers
displayer_source = Displayer()
displayer_matches = Displayer()

-- global zoom
zoom = 1

-- profiler
profiler = Profiler()

-- exec
function run()

   -- get live params
   profiler:start('global', 'fps')
   pk = widget.verticalSlider_1.value
   k = (widget.verticalSlider_2.value)/10 + 1

   profiler:start('get-frame')
   frame = source:forward()
   profiler:lap('get-frame')

   profiler:start('compute-sift')
   if sift_frames then
      if not sift_frames_prev then
         sift_frames_prev = torch.Tensor()
         sift_descs_prev = torch.Tensor()
         frame_prev = torch.Tensor()
      end
      frame_prev:resizeAs(frame):copy(frame)
      sift_frames_prev:resizeAs(sift_frames):copy(sift_frames)
      sift_descs_prev:resizeAs(sift_descs):copy(sift_descs)
   end
   sift_frames, sift_descs = vlfeat.sift{image=frame:select(3,2), edgeThresh=k, peakThresh=pk}
   if sift_frames_prev then
      matches = vlfeat.sift_match{descs1=sift_descs, descs2=sift_descs_prev, threshold=5}
   end
   profiler:lap('compute-sift')

   profiler:start('render-sift')
   sift_render = sift_render or torch.Tensor(1,1)
   vlfeat.sift_display{image=frame, frames=sift_frames, descriptors=descriptors,
                       totensor=sift_render}
   if matches then
      sift_match_render = sift_match_render or torch.Tensor()
      vlfeat.sift_match_display{images={frame, frame_prev}, 
                                matches=matches,
                                frames={sift_frames, sift_frames_prev},
                                totensor=sift_match_render}
   end
   profiler:lap('render-sift')

   -- display main image
   profiler:start('display')
   painter:gbegin()
   painter:showpage()
   displayer_source:show{tensor=sift_render, painter=painter, globalzoom=zoom,
                         min=0, max=1, offset_x=0, offset_y=20,
                         legend='SIFT features'}
   if sift_match_render then
      displayer_matches:show{tensor=sift_match_render, painter=painter, globalzoom=zoom,
                             min=0, max=1, offset_x=0, offset_y=40+sift_render:size(2),
                             legend='SIFT matches'}
   end
   profiler:lap('display')

   -- and params
   painter:setfont(qt.QFont{family='Papyrus',serif=false,bold=true,italic=false,size=12})
   painter:moveto(500, frame:size(2)*2*zoom + 80); painter:show('edgeThresh = ' .. k)
   painter:moveto(500, frame:size(2)*2*zoom + 100); painter:show('peakThresh = ' .. pk)

   --disp times
   profiler:lap('global')
   profiler:displayAll{painter=painter, 
                       x=10, y=frame:size(2)*2*zoom+80,
                       zoom=1/2}
   painter:gend()

   -- collect
   collectgarbage()
end

-- Loop Process
local timer = qt.QTimer()
timer.interval = 5
timer.singleShot = true
timer:start()
qt.connect(timer, 'timeout()', function() run() timer:start() end)

-- Start Process
widget.windowTitle = "Live SIFT Descriptors"
widget:show()
