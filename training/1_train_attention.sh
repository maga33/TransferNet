cd 1_train_attention 

## datasets
ln -s ../../data
## models
ln -s ../../models/pretrainedCNN.caffemodel
## create directories
ln -s ../../caffe
mkdir snapshot
mkdir training_log

## start training
#./start_train.sh

## copy trained model
