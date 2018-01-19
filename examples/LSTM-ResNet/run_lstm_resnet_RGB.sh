#!/bin/bash

TOOLS=../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

#GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB-3.prototxt -weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel  
GLOG_logtostderr=1 $TOOLS/caffe train --solver=./5-6resnet_50_solver.prototxt -weights ./ResNet-50-model.caffemodel
echo "Done."
