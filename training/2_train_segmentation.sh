cd 2_train_segmentation 

## datasets
ln -s ../../data
#$ copy the model
ln -s ../../models/attentionNet.caffemodel
## create directories
ln -s ../../caffe
mkdir snapshot
mkdir training_log

## start training
#./start_train.sh

## copy trained model
