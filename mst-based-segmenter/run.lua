#!/usr/bin/env torch
------------------------------------------------------------
-- a scene segmenter, base on a ConvNet trained end-to-end 
-- to predict class distributions at dense locations.
--
-- Clement Farabet
--

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'imgraph'
require 'nnx'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='camera index: /dev/videoIDX (if no video given)', 
          default=0}
op:option{'-v', '--video', action='store', dest='video',
          help='video file to process'}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=10}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=10}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}
op:option{'-z', '--zoom', action='store', dest='zoom',
          help='display zoom', default=1}
opt,args = op:parse()

if not opt.video then
   -- load camera
   require 'camera'
   video = image.Camera(opt.camidx, opt.width, opt.height)
else
   -- load video
   require 'ffmpeg'
   video = ffmpeg.Video{path=opt.video, width=opt.width, height=opt.height, 
                        fps=opt.fps, length=opt.seconds, delete=false}
end

-- setup GUI (external UI file)
if not win or not widget then 
   win = qtwidget.newwindow(opt.width*2*opt.zoom, opt.height*opt.zoom,
                            'A simple mst-based cartoonizer')
end

-- gaussian (a gaussian, really, is always useful)
gaussian = image.gaussian(3)

-- process function
function process()
   -- (1) grab frame
   frame = video:forward()

   -- (2) compute affinity graph on input image
   frame_smoothed = image.convolve(frame, gaussian, 'same')
   graph = imgraph.graph(frame_smoothed)

   -- (3) cut graph using min-spanning tree
   mstsegm = imgraph.segmentmst(graph, 2, 20)

   -- (4) pool the input frame into the segmentation
   cartoon = imgraph.histpooling(frame:clone(), mstsegm)
end

-- display function
function display()
   -- display input image + result
   image.display{image={frame,cartoon}, win=win, zoom=opt.zoom}
end

-- setup gui
timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              process()
              win:gbegin()
              win:showpage()
              display()
              win:gend()
              timer:start()
           end)
timer:start()
