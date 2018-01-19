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

    alexnet model:（https://people.eecs.berkeley.edu/~lisa_anne/single_frame_all_layers_hyb_RGB_iter_5000.caffemodel）, put it in $ROOT/examples/LSTM-AlexNet
    
    googlenet model:（https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet）, put it in $ROOT/examples/LSTM-GoogLeNet

    ResNet-50 model:（https://github.com/KaimingHe/deep-residual-networks#models）, put it in $ROOT/examples/LSTM-ResNet
* Run the run_lstm_alexnet_RGB.sh in in the terminal window to train the lstm-alexnet model
```
cd $ROOT/examples/LSTM-AlexNet
sh run_lstm_alexnet_RGB.sh
```
* Run the run_lstm_googlenet_RGB.sh in the terminal window to train the lstm-googlenet model
```
cd $ROOT/examples/LSTM-GoogLeNet
sh run_lstm_googlenet_RGB.sh
```
* Run the run_lstm_resnet_RGB.sh in the terminal window to train the lstm-resNet model
```
cd $ROOT/examples/LSTM-ResNet
sh run_lstm_resnet_RGB.sh
```
### Test

The test code is in $ROOT/examples/test

* Run 5-6classify_video-alexnet.py to test: in python terminal. 


