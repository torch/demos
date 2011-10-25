#!/usr/bin/env qlua

------------------------------------------------------------
-- computes disparities from a pair of stereo images
--

require 'XLearn'
stereo = xrequire 'stereo'

if not stereo then
   error('please install stereo package (make USE_3RDPARTY=1)')
end

stereo.testme()

print([[

To know more about these algos, start a Lua shell, and try:
> require 'stereo'
> left = image.load('/path/to/left.jpg')
> right = image.load('/path/to/right.jpg')
> stereo.infer()
... prints some online help on the function ...
> result = stereo.infer{left=left, right=right}
> image.display(result)
]])
