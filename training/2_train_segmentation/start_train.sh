LOGDIR=./training_log/
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=./attentionNet.caffemodel # pretrained model in imagenet
#WEIGHTS=./snapshot/070_iter_6000.caffemodel # finetuned model in voc

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS 

