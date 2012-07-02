
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

ADD_TORCH_DOK(dok/supervised tutorials_supervised "Tutorials" "Supervised Learning" 1.0)
ADD_TORCH_DOK(dok/unsupervised tutorials_unsupervised "Tutorials" "Unsupervised Learning" 1.1)
ADD_TORCH_DOK(dok/graphicalmodels tutorials_graphical "Tutorials" "Graphical Models" 1.2)
ADD_TORCH_DOK(dok/gui tutorials_gui "Tutorials" "Graphical User Interfaces" 1.3)
