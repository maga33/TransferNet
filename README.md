## TransferNet: Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network

Created by [Seunghoon Hong](http://cvlab.postech.ac.kr/~maga33/), [Junhyuk Oh](https://sites.google.com/a/umich.edu/junhyuk-oh/), [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) and [Honglak Lee](http://web.eecs.umich.edu/~honglak/)


Project page: [http://cvlab.postech.ac.kr/research/dppnet/]

## Introduction

This repository contains the source code for the semantic segmentation algorithm described in the following paper:   
* Seunghoon Hong, Junhyuk Oh, Bohyung Han, Honglak Lee, **"Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network**"
    _In IEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016.

```
@inproceedings{HongOHL2016,
  title={Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network},
  author={Hong, Seunghoon and Oh, Junhyuk and Han, Bohyung and Lee, Honglak},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
  year={2016}
}
```

Pleae refer to our [arXiv tech report](http://arxiv.org/abs/1512.07928) for details. 

## Installation
You need to compile the modified Caffe library in this repository.
Please consult following [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html). 
After installing rquired libraries for Caffe, you should compile both Caffe and its matlab interface as follows 

```
cd caffe
make all
make matcaffe
```

## Training


## Inference


### Licence

This software is for research purpose only.
Check LICENSE file for details.

