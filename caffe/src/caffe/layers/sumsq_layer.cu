#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumSqForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, 
    const int height, const int width, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height ;
    int h = (index / width) % height;
    int w = index % width;
    int dim = height * width;

    int bottom_index = n*width*height*channels + h*width + w;
    top_data[index]=0;
    for (int c=0; c< channels; c++){
	top_data[index] += bottom_data[bottom_index]*bottom_data[bottom_index];
	bottom_index += dim;
    }
  }
}

    
template <typename Dtype>
__global__ void SumSqBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_data, const int num, const int channels, 
    const int height, const int width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height / channels;
    int c = (index / width / height) % channels;
    int h = (index / width) % height;
    int w = index % width;
    int dim = height * width;

    int top_index = n*width*height + h*width + w;
    bottom_diff[index] = top_diff[top_index] * 2.0*bottom_data[index];
  }
}



template <typename Dtype>
void SumSqLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // net_->ForwardPrefilled();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = top[0]->height();
    int width = top[0]->width();

    const Dtype* X = bottom[0]->gpu_data();
    Dtype* Y = top[0]->mutable_gpu_data();

    SumSqForward<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>(
        num*height*width, X, num, channels,
        height, width, Y);
    

}

template <typename Dtype>
void SumSqLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    int count = bottom[0]->count();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
 
    SumSqBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, num, channels,
        height, width, bottom_diff);

}

INSTANTIATE_LAYER_GPU_FUNCS(SumSqLayer);

}
