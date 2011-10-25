#!/usr/bin/env qlua

------------------------------------------------------------
-- computes the optical flow-field of a pair of images
--

require 'XLearn'
opencv = xrequire 'opencv'
oflow = xrequire 'opticalFlow'

if opencv then
   opencv.test_calcOpticalFlow()
else
   print('please install opencv wrapper to see more (make USE_OPENCV=1)')
end

if oflow then

   oflow.testme()

   print([[

To know more about these algos, start a Lua shell, and try:
> require 'opticalFlow'
> img1 = image.load('/path/to/img1.jpg')
> img2 = image.load('/path/to/img2.jpg')
> opticalFlow.infer()
... prints some online help on the function ...
> flow_x, flow_y = opticalFlow.infer{pair={img1,img2}}
> image.displayList{images={flow_x,flow_y}}
... the flow can be displayed in HSL space ...
> hsl = opticalFlow.field2rgb(flow_x, flow_y)
> image.display(hsl)
]])
else
   print('please install 3rd party packages to see more (make USE_3RDPARTY=1)')
end

