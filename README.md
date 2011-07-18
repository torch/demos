# Demos & Turorials for Torch7.

All the demos/tutorials provided in this package
require the following dependencies to be installed, in
order to work.

## Install dependencies on Linux (Ubuntu > 9.04):

1/ Basic tools

``` sh
$ apt-get install gcc g++ git libreadline5-dev cmake wget
```

2/ QT4 (at least 4.4)

``` sh
$ apt-get install libqt4-core libqt4-gui libqt4-dev
```

3/ Extras

``` sh
$ apt-get install ffmpeg gnuplot
```

## Install dependencies on Mac OS X > 10.5:

0/ Install the dev tools (gcc/g++ from Apple),
   and we highly recommend to get Homebrew
   (http://mxcl.github.com/homebrew/) as a replacement
   for MacPorts.

1/ Basic tools, using Homebrew:

``` sh
$ brew install git readline cmake wget
```

2/ Install QT4 (at least 4.4)

``` sh
$ brew install qt
```

3/ Extras

``` sh
$ brew install ffmpeg gnuplot
```

## Install Lua, Luarocks and Torch7 on both platforms

1/ Lua 5.1 + Luarocks + xLua 

``` sh
$ git clone https://github.com/clementfarabet/lua4torch
$ cd lua4torch
$ make install PREFIX=/usr/local
```

2/ Torch7 (a numeric package for Lua)

``` sh
$ luarocks install torch
```

At this stage, extra packages will be auto-installed by
each demo/script, as needed. In case something goes wrong, 
each extra package can be installed like Torch:

3/ Extra packages

``` sh
$ luarocks install image    # an image library for Torch7
$ luarocks install nnx      # lots of extra neural-net modules
$ luarocks install camera   # a camera interface for Linux/MacOS
$ luarocks install ffmpeg   # a video decoder for most formats
```
