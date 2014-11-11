# Demos & Turorials for Torch7.

All the demos/tutorials provided in this repo require Torch7 to be installed,
as well as some extra (3rd-party) packages.

## Install

### Torch7

Follow instructions on: [Torch7's homepage](http://www.torch.ch/).

### 3rd-party packages

Different demos/tutorials rely on different 3rd-party packages. If a demo
crashes because it can't find a package then simply try to install it using 
luarocks:

``` sh
$ luarocks install image    # an image library for Torch7
$ luarocks install nnx      # lots of extra neural-net modules
$ luarocks install camera   # a camera interface for Linux/MacOS
$ luarocks install ffmpeg   # a video decoder for most formats
$ ...
```

A complete list of packages can be obtained by doing:

``` sh
$ luarocks list
```

or checking out [this page](http://htmlpreview.github.io/?https://github.com/torch/rocks/blob/master/index.html).

## Tutorials

https://github.com/torch/tutorials

## Credits

These demos were slowly put together by: Clement Farabet & Roy Lowrance and e-lab
