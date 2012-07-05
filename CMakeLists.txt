
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

ADD_TORCH_DOK(dok/supervised tutorials_supervised "Tutorials" "Supervised Learning" 1.0)
ADD_TORCH_DOK(dok/unsupervised tutorials_unsupervised "Tutorials" "Unsupervised Learning" 1.1)
ADD_TORCH_DOK(dok/graphicalmodels tutorials_graphical "Tutorials" "Graphical Models" 1.2)
ADD_TORCH_DOK(dok/gui tutorials_gui "Tutorials" "Graphical User Interfaces" 1.3)

SET(luasrc linear-regression/example-linear-regression.lua
		   logistic-regression/example-logistic-regression.lua
		   logistic-regression/example-logistic-regression.csv
		   train-a-digit-classifier/train-on-mnist.lua
		   train-a-digit-classifier/dataset-mnist.lua
		   train-on-cifar/train-on-cifar.lua
		   train-on-housenumbers/train-on-housenumbers.lua
		   train-autoencoder/train-autoencoder.lua)

ADD_TORCH_PACKAGE(tutorials "" "${luasrc}" "Tutorials")
