# Tracking with Deep Neural Networks

This script provides a real-time tracking with a video sequence
from a camera or datasets (presented at CISS 2013)

## Install

To run this program, you will need to install Torch7 and 'nnx' package
In addition to the usual (torch, nnx, etc.) this demo requires the 
following modules to be installed via luarocks:

``` sh
neuflow
inline
image
```

## Demonstration

For the demonstration, use the following commands:

``` sh
$ torch run.lua        # run on computer
$ torch run.lua -nf    # run on neuFlow
```
