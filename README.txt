
============================================================
Demos & Turorials for Torch7.

All the demos/tutorials provided in this package
require the following dependencies to be installed, in
order to work.

============================================================
Linux (Ubuntu > 9.04):

1/ Basic tools
$ apt-get install gcc g++ git libreadline5-dev cmake wget

2/ QT4 (at least 4.4)
$ apt-get install libqt4-core libqt4-gui libqt4-dev

3/ Extras
$ apt-get install ffmpeg gnuplot

============================================================
Mac OS X > 10.5:

0/ Install the dev tools (gcc/g++ from Apple),
   and we highly recommend to get Homebrew
   (http://mxcl.github.com/homebrew/) as a replacement
   for MacPorts.

1/ Basic tools, using Homebrew:
$ brew install git readline cmake wget

2/ Install QT4 (at least 4.4)
$ brew install qt

3/ Extras
$ brew install ffmpeg gnuplot

============================================================
Linux and Mac OS X

1/ Lua 5.1 + Luarocks + xLua 
$ git clone https://github.com/clementfarabet/lua4torch
$ cd lua4torch
$ make install PREFIX=/usr/local

2/ Torch7 (a numeric package for Lua)
$ luarocks install torch

At this stage, extra packages will be auto-installed by
each demo/script, as needed. In case something goes wrong, 
each extra package can be installed like Torch:

3/ Extra packages
$ luarocks install image    # an image library for Torch7
$ luarocks install nnx      # lots of extra neural-net modules
$ luarocks install camera   # a camera interface for Linux/MacOS
$ luarocks install ffmpeg   # a video decoder for most formats
