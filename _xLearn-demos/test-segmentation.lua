#!/usr/bin/env qlua

------------------------------------------------------------
-- tests a few segmentation algorithms available
-- in the repo
--

require 'XLearn'
mstsegm = xrequire 'mstsegm'
powerwatersegm = xrequire 'powerwatersegm'

if not mstsegm or not powerwatersegm then
   error('please install 3rd party packages (make USE_3RDPARTY=1)')
end

mstsegm.testme()
powerwatersegm.testme()

print([[

To know more about these algos, start a Lua shell, and try:
> require 'mstsegm'
> img = image.load('/path/to/image.jpg')
> mstsegm.infer()
... prints some online help on the function ...
> result = mstsegm.infer{image=img} 
> image.display(result)
> require 'powerwatersegm'
> powerwatersegm.infer() 
]])
