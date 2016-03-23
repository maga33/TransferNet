LOGDIR=./training_log/
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
#WEIGHTS=./clsnset_voc_224x224_no_bn.caffemodel # finetuned model in voc
WEIGHTS=./pretrainedCNN.caffemodel # pretrained model in imagenet

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS 

