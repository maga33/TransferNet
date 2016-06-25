## TransferNet: Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network

Created by [Seunghoon Hong](http://cvlab.postech.ac.kr/~maga33/), [Junhyuk Oh](https://sites.google.com/a/umich.edu/junhyuk-oh/), [Honglak Lee](http://web.eecs.umich.edu/~honglak/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/)


Project page: [http://cvlab.postech.ac.kr/research/transfernet/]

## Introduction

This repository contains the source code for the semantic segmentation algorithm described in the following paper:   
* Seunghoon Hong, Junhyuk Oh, Honglak Lee, Bohyung Han, **"Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network**"
    _In IEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016.

```
@inproceedings{HongOLH2016,
  title={Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network},
  author={Hong, Seunghoon and Oh, Junhyuk and Lee, Honglak and Han, Bohyung},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
  year={2016}
}
```

Pleae refer to our [arXiv tech report](http://arxiv.org/abs/1512.07928) for details. 

## Installation

You need to compile the modified Caffe library in this repository.
Please consult [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html) for details. 
After installing rquired libraries for Caffe, you need to compile both Caffe and its Matlab interface as follows: 

```
cd caffe
make all
make matcaffe
```

After installing Caffe, you can download datasets, pre-trained models, and other libraries by following script:

```
setup.sh
```


## Training

Training procedures are composed of two steps, which are implemented in different directories:
  * `training/1_train_attention` : pre-train attention and classification network with image-level class labels.
  * `training/2_train_segmentation` : train entire network including a decoder with pixel-wise class labels. 

You can run training with following scripts

```
cd training
./1_train_attention.sh
./2_train_segmentation.sh
```


## Inference

You can run inference on PASCAL VOC 2012 validatoin images using the trained model as follow:

```
cd inference
matlab -nodesktop -r run_inference
```

By default, this script will perform an inference on PASCAL VOC 2012 validation images using the pre-trained model.
You may need to modify the code if you want to apply the model to different dataset or use the different models. 

### Licence

This software is for research purpose only.
Check LICENSE file for details.

