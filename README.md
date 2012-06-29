# Demos & Turorials for Torch7.

All the demos/tutorials provided in this repo require Torch7 to be installed,
as well as some extra (3rd-party) packages.

## Install

### Torch7
Follow instructions on: [Torch7's homepage](http://www.torch.ch/).

### 3rd-party packages

Different demos/tutorials rely on different 3rd-party packages. If a demo
crashes because it can't find a package then simply try to install it using 
torch-pkg:

``` sh
$ torch-pkg install image    # an image library for Torch7
$ torch-pkg install nnx      # lots of extra neural-net modules
$ torch-pkg install camera   # a camera interface for Linux/MacOS
$ torch-pkg install ffmpeg   # a video decoder for most formats
$ ...
```

A complete list of packages can be obtained by doing:

``` sh
$ torch-pkg list
```

or checking out [this page](http://data.neuflow.org/torch).

## Tutorials

Each directory provides a tutorial or a demo, with no particular order.
It's a good idea to have [Torch's documentation](http://www.torch.ch/)
open on the side, for reference. As Torch is based on the Lua language,
it's also a good idea to go through the [Lua 5.1](http://www.lua.org/manual/5.1/)
book.

## Credits

These demos were slowly put together by: Clement Farabet & Roy Lowrance.
