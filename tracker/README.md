# Tracking with Deep Neural Networks

This script provides a real-time tracking with a video sequence
from a camera or datasets (presented at CISS 2013)

## Install

To run this program, you will need to install Torch7 and 'nnx' package
In addition to the usual (torch, nnx, etc.) this demo requires the 
following modules to be installed via luarocks:

``` sh
image
ffmpeg
```

## Demonstration
Only for the first time run: 

```sh
 ./compile.sh
```

For the demonstration:

``` sh
qlua run.lua
```
