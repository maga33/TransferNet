#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ChannelSumForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* scale_data, const int num, const int channels, 
    const int height, const int width, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels ;
    int c = index % channels;
    int dim = height * width;

    int bottom_index = n*width*height*channels + c*width*height;
    top_data[index]=0;
    for (int c=0; c< dim; ++c){
	top_data[index] += bottom_data[bottom_index]*scale_data[index];
	bottom_index += 1;
    }
  }
}

    
template <typename Dtype>
__global__ void ChannelSumBackwardData(const int nthreads, const Dtype* top_diff,
    const Dtype* scale_data, const int num, const int channels, 
    const int height, const int width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height / channels;
    int c = (index / width / height) % channels;

    int w_index = n*channels + c;
    bottom_diff[index] = top_diff[w_index] * scale_data[w_index];
  }
}


template <typename Dtype>
__global__ void ChannelSumBackwardScale(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_data, const int num, const int channels, 
    const int height, const int width, Dtype* scale_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    int c = index % channels;
    int dim = height * width;

    int bottom_index = n*width*height*channels + c*height*width;
    scale_diff[index]=0;
    for (int i=0; i <dim; i++){
	scale_diff[index] += top_diff[index] * bottom_data[bottom_index];
        bottom_index += 1; 
    }
  }
}

template <typename Dtype>
void ChannelSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // net_->ForwardPrefilled();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();

    const Dtype* X = bottom[0]->gpu_data();
    const Dtype* W = bottom[1]->gpu_data();
    Dtype* Y = top[0]->mutable_gpu_data();

    ChannelSumForward<Dtype><<<CAFFE_GET_BLOCKS(num*channels), CAFFE_CUDA_NUM_THREADS>>>(
        num*channels, X, W, bottom[0]->num(), channels,
        height, width, Y);
    

    /*
    Dtype* Y = top[0]->mutable_gpu_data();
    caffe_gpu_set(top[0]->count(), (Dtype)0., Y);
    Dtype* tmp = tmp_.mutable_gpu_data();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_gpu_scale(dim, W[j], X, tmp);
        caffe_gpu_add(dim, tmp, Y, Y);
        X += bottom[0]->count(2);
      }
      Y += dim;
      W += channels;
    }
   */
}

template <typename Dtype>
void ChannelSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    int count = bottom[0]->count();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* scale_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* scale_diff = bottom[1]->mutable_gpu_diff();
 
    ChannelSumBackwardData<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scale_data, num, channels,
        height, width, bottom_diff);

   
    ChannelSumBackwardScale<Dtype><<<CAFFE_GET_BLOCKS(num*channels), CAFFE_CUDA_NUM_THREADS>>>(
        num*channels, top_diff, bottom_data, num, channels,
        height, width, scale_diff);


/*
    Dtype* Y_diff_X = Y_diff_X_.mutable_gpu_data();
    Dtype* tmp = tmp_.mutable_gpu_data();
    caffe_gpu_set(bottom[1]->count(), (Dtype)0., W_diff);

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_gpu_scale(dim, W[j], Y_diff, X_diff);
        caffe_gpu_mul(dim, Y_diff, X, Y_diff_X);
	caffe_gpu_asum(dim, Y_diff_X, W_diff); //very suspicous

        X_diff += dim;
        Y_diff_X += dim;
        W_diff += 1;
        X += dim;
      }
      Y_diff += dim;
      W += channels;
    }

*/
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelSumLayer);

}
