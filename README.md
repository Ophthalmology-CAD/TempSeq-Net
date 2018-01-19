## Usage Instructions for TempSeq-Net:
* Clone this repository somewhere, let's refer to it as $ROOT
```
git clone https://github.com/Ophthalmology-CAD/TempSeq-Net.git
```
### Train
* Compile the caffe and pycaffe.
```
cd $ROOT
make all 
make test 
make runtest 
make pycaffe
```
* Download the pre-trained model. 

    alexnet model:（https://people.eecs.berkeley.edu/~lisa_anne/single_frame_all_layers_hyb_RGB_iter_5000.caffemodel）, put it in $ROOT/examples/AlexNet
    
    googlenet model:（https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet）, put it in $ROOT/examples/GoogLeNet

    ResNet-50 model:（https://github.com/KaimingHe/deep-residual-networks#models）, put it in $ROOT/examples/ResNet
* Run the run_alexnet_lstm_RGB.sh in in the terminal window to train the alexnet model
```
cd $ROOT
sh examples/AlexNet/run_alexnet_lstm_RGB.sh
```
* Run the run_googlenet_lstm_RGB.sh in the terminal window to train the googlenet model
```
cd $ROOT
sh examples/GoogLeNet/run_googlenet_lstm_RGB.sh
```
* Run the run_resnet_lstm_RGB.sh in the terminal window to train the ResNet model
```
cd $ROOT
sh examples/ResNet/run_resnet_lstm_RGB.sh
```
### Test

The test code is in $ROOT/examples/test

* Run 5-6classify_video-alexnet.py to test: in python terminal. 


