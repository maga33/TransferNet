#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* scale_data, const int num, const int channels, 
    const int height, const int width, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height / channels;
    int h = (index / width) % height;
    int w = index % width;
    int dim = height * width;
    int scale_idx = n * dim + h * width + w;
    top_data[index] = bottom_data[index] * scale_data[scale_idx];
  }
}

template <typename Dtype>
__global__ void ScaleBackwardData(const int nthreads, const Dtype* top_diff,
    const Dtype* scale_data, const int num, const int channels, 
    const int height, const int width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height / channels;
    int h = (index / width) % height;
    int w = index % width;
    int dim = height * width;
    int scale_idx = n * dim + h * width + w;
    bottom_diff[index] = top_diff[index] * scale_data[scale_idx];
  }
}

template <typename Dtype>
__global__ void ScaleBackwardScale(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_data, const int num, const int channels, 
    const int height, const int width, Dtype* scale_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height;
    int h = (index / width) % height;
    int w = index % width;
    int dim = height * width;

    int top_idx = n * channels * height * width + h * width + w;
    scale_diff[index] = 0;
    for (int c = 0; c < channels; ++c) {
      scale_diff[index] += top_diff[top_idx] * bottom_data[top_idx];
      top_idx += dim;
    }
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (scale_by_param_) {
    for (int i = 0; i < top.size(); ++i) {
      caffe_gpu_scale(top[i]->count(), factor_, bottom[i]->gpu_data(), 
        top[i]->mutable_gpu_data());
    }
  }
  else { 
    int count = top[0]->count();
    int channels = top[0]->channels();
    int height = top[0]->height();
    int width = top[0]->width();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* scale_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    
   ScaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, bottom[0]->num(), channels,
        height, width, top_data);
    
    // net_->ForwardPrefilled();
    /*
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);
    const Dtype* X = bottom[0]->gpu_data();
    const Dtype* W = bottom[1]->gpu_data();
    Dtype* Y = top[0]->mutable_gpu_data();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_gpu_mul(dim, X, W, Y);
        Y += top[0]->count(2);
        X += bottom[0]->count(2);
      }
      W += dim;
    }
    */
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (scale_by_param_) {
    for (int i = 0; i < top.size(); ++i) {
      caffe_gpu_scale(top[i]->count(), factor_, top[i]->gpu_diff(), 
        bottom[i]->mutable_gpu_diff());
    } 
  }
  else {
    int count = top[0]->count();
    int num = top[0]->num();
    int channels = top[0]->channels();
    int height = top[0]->height();
    int width = top[0]->width();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* scale_data = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* scale_diff = bottom[1]->mutable_gpu_diff();

    ScaleBackwardData<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scale_data, num, channels,
        height, width, bottom_diff);

    ScaleBackwardScale<Dtype><<<CAFFE_GET_BLOCKS(num * height * width), 
        CAFFE_CUDA_NUM_THREADS>>>(
          num * height * width, top_diff, bottom_data, num, channels,
          height, width, scale_diff);

    //net_->Backward();
    /*
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);
    const Dtype* X = bottom[0]->gpu_data();
    const Dtype* W = bottom[1]->gpu_data();
    const Dtype* Y_diff = top[0]->gpu_diff();
    Dtype* X_diff = bottom[0]->mutable_gpu_diff();
    Dtype* W_diff = bottom[1]->mutable_gpu_diff();
    Dtype* Y_diff_X = Y_diff_X_.mutable_gpu_data();
    caffe_gpu_mul(Y_diff_X_.count(), Y_diff, X, Y_diff_X);
    caffe_gpu_set(bottom[1]->count(), (Dtype)0., W_diff);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_gpu_mul(dim, Y_diff, W, X_diff);
        caffe_gpu_add(dim, Y_diff_X, W_diff, W_diff);
        
        X_diff += dim;
        Y_diff += dim;
        Y_diff_X += dim;
      }
      X += bottom[0]->count(1);
      W += bottom[1]->count(1);
      W_diff += dim;
    }
    */
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);

}  // namespace caffe
