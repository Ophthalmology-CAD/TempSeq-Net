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

    pre-trained model:（https://people.eecs.berkeley.edu/~lisa_anne/single_frame_all_layers_hyb_RGB_iter_5000.caffemodel）, put it in $ROOT/examples
* Run the 5-6run_lstm_RGB.sh in in the terminal window to train the model
```
cd $ROOT/examples
sh 5-6run_lstm_RGB.sh
```
### Test

The test code is in $ROOT/examples/test

* Run 5-6classify_video-alexnet.py to test: in python terminal. 


