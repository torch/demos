#!/usr/bin/env torch
-------------------------------------------------------------------
-- Tracking with Deep Neural Networks (Presented at CISS 2013)
------------------------------------------------------------------- 
-- Deep neural networks models are applied to long-term
-- tracking objects. This script provides a real-time tracking
-- with a given video sequence.
--
-- the idea, adapted from LeCun/Kavukcuoglu's
-- original demo (presented during ICCV'09), is as follows:
--
-- + A large feature extractor is used
--   to build up a representation that is more compact
--   and robust to rotation/distortion/translation than the
--   original pixel space
--
-- + The feature extractor is augmented with
--   a radial basis function and a confidence map is estimated
--
-- + The parallel structure of this model can be accelerated
--   by neuFlow, our custom hardware
--
-- Copyright: Jonghoon Jin, Eugenio Culurciello,
--            Clement Farabet, Yann LeCun
-------------------------------------------------------------------

require 'nnx'

title = 'Tracking with Deep Neural Networks'

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-t', '--threads', action='store', dest='threads',
          help='set the number of threads', default=8}
op:option{'-s', '--source', action='store', dest='source',
          help='image source, can be one of: camera | video | dataset', default='camera'}
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, specify the camera index: /dev/videoIDX', default=0}
op:option{'-S', '--save', action='store', dest='save',
          help='path to save video stream'}
op:option{'-v', '--video', action='store', dest='video',
          help='path to video', default=''}
op:option{'-r', '--vrate', action='store', dest='fps',
          help='video rate (fps), for video only', default=5}
op:option{'-l', '--vlength', action='store', dest='length',
          help='video length (seconds), for video only', default=10}
op:option{'-p', '--dspath', action='store', dest='dspath',
          help='path to dataset', default=''}
op:option{'-n', '--dsencoding', action='store', dest='dsencoding',
          help='dataset image format', default='jpg'}
op:option{'-O', '--dsoutput', action='store', dest='dsoutput',
          help='file to save tracker output to, for dataset only'}
op:option{'-N', '--nogui', action='store_true', dest='nogui',
          help='turn off the GUI display (only useful with dataset)'}
op:option{'-e', '--encoder', action='store', dest='encoder',
          help='can be: random | supervised | unsupervised', default='unsupervised.net'}
op:option{'-w', '--width', action='store', dest='width',
          help='detection window width', default=640}
op:option{'-h', '--height', action='store', dest='height',
          help='detection window height', default=480}
op:option{'-b', '--box', action='store', dest='box',
          help='box (training) size', default=128}
op:option{'-d', '--downsampling', action='store', dest='downs',
          help='downsampling ratio(-1 to downsample as much as possible)',default=2}
op:option{'-f', '--file', action='store', dest='file',
          help='file to sync memory to', default='memory'}
op:option{'-th','--threshold', action='store', dest='threshold',
          help='set a threshold for detection', default=0.7}
op:option{'-st','--std', action='store', dest='std',
          help='set a gaussian spread for RBF', default=0.5}

options,args = op:parse()
options.classes = {'Object 1'} -- class names

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(options.threads)
print('Number of threads:', torch.getnumthreads())

print('')
print('e-Lab ' .. title) 
print('loading encoder:')
print('')

-- load required submodules
state = require 'state'
source = require 'source'
process = require 'process'

-- load gui and display routine, if necessary
if not options.nogui then
   -- setup GUI (external UI file)
   require 'qt'
   require 'qtwidget'
   require 'qtuiloader'
   widget  = qtuiloader.load('g.ui')
   painter = qt.QtLuaPainter(widget.frame)
   display = require 'display'
   ui      = require 'ui'
end

-- save ?
if options.save then
   os.execute('mkdir -p ' .. options.save)
end
-- for memory files
sys.execute('mkdir -p scratch')

state.begin()
