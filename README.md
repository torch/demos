# Demos & Turorials for Torch7.

All the demos/tutorials provided in this package
require the following dependencies to be installed, in
order to work.

## Install dependencies on Linux (Ubuntu > 9.04):

1/ Basic tools

``` sh
$ apt-get install gcc g++ git libreadline-dev cmake wget
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

## Install Torch7 (full instructions on torch.ch) and extra packages

``` sh
$ git clone git://github.com/andresy/torch.git
$ cd torch
$ mkdir build; cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
$ make install
```

``` sh
$ torch-pkg install image    # an image library for Torch7
$ torch-pkg install nnx      # lots of extra neural-net modules
$ torch-pkg install camera   # a camera interface for Linux/MacOS
$ torch-pkg install ffmpeg   # a video decoder for most formats
```
