#!/usr/bin/env qlua
----------------------------------------------------------------------
-- WHAT: segments a image source.
--       the image source must be provided on the command line.
--
require 'XLearn'
require 'os'
require 'mstsegm'
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
widget = qtuiloader.load('live-segm+sift.ui')
painter = qt.QtLuaPainter(widget.frame)

-- video source
source = nn.ImageSource{type = options.source or 'camera', 
                        path = options.path,
                        cam_idx = options.camidx,
                        fps = 20,
                        length = 10,
                        width = 320,
                        height = 240}

-- displayers
displayer_source = Displayer()
displayer_segments = Displayer()
displayer_matches = Displayer()

-- global zoom
zoom = 1

-- cheap segm ?
cheap = false

-- profiler
profiler = Profiler()

-- incremental segmentation
do
   local nslices = 8
   local cheap = false
   local segmbuf
   local ringbuf = {}
   local ringbuf_i = 0

   -- incremental segm function
   function segm(img)
      -- new frame
      ringbuf_i = ringbuf_i + 1

      -- resize buffer
      segmbuf = segmbuf or torch.Tensor(img:size(1), img:size(2), img:size(3), nslices)
      -- store frame
      ringbuf[ringbuf_i] = torch.Tensor():resizeAs(img):copy(img)

      -- cleanup
      ringbuf[ringbuf_i-nslices] = nil
      collectgarbage()

      -- compute segm
      local segments
      if ringbuf_i > nslices then
         -- fill buffer
         for i = 1,nslices do
            segmbuf:select(4,i):copy(ringbuf[ringbuf_i-nslices+i])
         end

         -- compute SIFT first
         local frame = segmbuf:select(4,nslices)
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
         sift_frames, sift_descs = vlfeat.sift{image=frame:select(3,2), edgeThresh=kk, peakThresh=pk}
         if sift_frames_prev then
            matches = vlfeat.sift_match{descs1=sift_descs, descs2=sift_descs_prev, threshold=5}
            if matches and matches:nDimension() == 2 then
               cleanmatches = cleanmatches or torch.Tensor()
               cleanmatches:resize(2,2,matches:size(2))
               for n = 1,matches:size(2) do
                  local idx = matches[1][n]
                  local idxp = matches[2][n]
                  cleanmatches[1][1][n] = sift_frames[1][idx]
                  cleanmatches[2][1][n] = sift_frames[2][idx]
                  cleanmatches[1][2][n] = sift_frames_prev[1][idxp]
                  cleanmatches[2][2][n] = sift_frames_prev[2][idxp]
               end
            else
               cleanmatches = nil
            end
         end

         -- segm
         if cheap then
            segments = mstsegm.infer{image=segmbuf:select(4,nslices), 
                                     connex=4, 
                                     k=k, min=minsize,
                                     incremental_cheap=true,
                                     matches=cleanmatches}
            return segments
         else
            segments = mstsegm.infer{image=segmbuf, 
                                     connex=4, 
                                     k=k, min=minsize,
                                     incremental=true,
                                     matches=cleanmatches}
            return segments:select(4,segments:size(4))
         end
      else
         -- in the mean time, dont do anything
         return img
      end
   end
end

-- exec
function run()

   -- get params
   minsize = widget.verticalSlider_1.value
   k = widget.verticalSlider_2.value
   kk = (widget.verticalSlider_3.value)/10 + 1
   pk = widget.verticalSlider_4.value

   profiler:start('global', 'fps')
   profiler:start('get-frame')
   frame = source:forward()
   profiler:lap('get-frame')

   profiler:start('get-segments')
   segments = segm(frame)
   profiler:lap('get-segments')

   profiler:start('render-sift')
   if matches then
      sift_match_render = sift_match_render or torch.Tensor()
      vlfeat.sift_match_display{images={frame, frame_prev}, 
                                matches=matches,
                                frames={sift_frames, sift_frames_prev},
                                totensor=sift_match_render}
   end
   profiler:lap('render-sift')

   painter:gbegin()
   painter:showpage()
   profiler:start('display')
   
   displayer_source:show{tensor = frame:add(segments), painter = painter, globalzoom=zoom, 
                         min=0,max=2, offset_x=0, offset_y=20, 
                         legend='camera image'}
   
   displayer_segments:show{tensor = segments, painter = painter, globalzoom=zoom, 
                           min = 0, max = 1, 
                           offset_x = frame:size(1), offset_y = 20, 
                           legend = 'segmented image'}

   if sift_match_render then
      displayer_matches:show{tensor=sift_match_render, painter=painter, globalzoom=zoom,
                             min=0, max=1, offset_x=0, offset_y=40+frame:size(2),
                             legend='SIFT matches'}
   end

   -- and params
   painter:setfont(qt.QFont{serif=false,bold=true,italic=false,size=12})
   painter:moveto(500, frame:size(2)*2*zoom + 80); painter:show('edgeThresh = ' .. kk)
   painter:moveto(500, frame:size(2)*2*zoom + 100); painter:show('peakThresh = ' .. pk)
   painter:moveto(500, frame:size(2)*2*zoom + 120); painter:show('k = ' .. k)
   painter:moveto(500, frame:size(2)*2*zoom + 140); painter:show('min = ' .. minsize)
   profiler:lap('display')

   --disp times
   profiler:lap('global')
   profiler:displayAll{painter = painter, x = 10, y = frame:size(2)*2*zoom + 80, zoom=1/2}

   painter:gend()
end

-- Loop Process
local timer = qt.QTimer()
timer.interval = 5
timer.singleShot = true
timer:start()
qt.connect(timer, 'timeout()', function() run() timer:start() end)

-- Start Process   
widget.windowTitle = "Live Segm"
widget:show()
