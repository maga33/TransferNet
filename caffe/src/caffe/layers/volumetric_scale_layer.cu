#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* Wc, const Dtype* Ws, const int num, const int channels, 
    const int height, const int width, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height / channels;
    int c = (index / (height * width)) % channels;
    int h = (index / width) % height;
    int w = index % width;
    int c_idx = n * channels + c;
    int s_idx = n * height * width + h * width + w;
    top_data[index] = bottom_data[index] * Wc[c_idx] * Ws[s_idx];
  }
}

template <typename Dtype>
__global__ void ScaleBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* Wc, const Dtype* Ws, const int num, const int channels, 
    const int height, const int width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height / channels;
    int c = (index / (height * width)) % channels;
    int h = (index / width) % height;
    int w = index % width;
    int c_idx = n * channels + c;
    int s_idx = n * height * width + h * width + w;
    bottom_diff[index] = top_diff[index] * Wc[c_idx] * Ws[s_idx];
  }
}

template <typename Dtype>
__global__ void ScaleBackwardChannelWeight(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_data, const Dtype* Ws, const int num, const int channels, 
    const int height, const int width, Dtype* Wc_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    int c = index % channels;
    int dim = height * width;
    
    int top_idx = n * channels * height * width + c * height * width;
    int s_idx = n * dim;
    Wc_diff[index] = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        Wc_diff[index] += top_diff[top_idx] * bottom_data[top_idx] * Ws[s_idx];
        top_idx++;
        s_idx++;
      }
    }
  }
}

template <typename Dtype>
__global__ void ScaleBackwardSpatialWeight(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_data, const Dtype* Wc, const int num, const int channels, 
    const int height, const int width, Dtype* Ws_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height;
    int h = (index / width) % height;
    int w = index % width;
    int dim = height * width;

    int top_idx = n * channels * height * width + h * width + w;
    int c_idx = n * channels;
    Ws_diff[index] = 0;
    for (int c = 0; c < channels; ++c) {
      Ws_diff[index] += top_diff[top_idx] * bottom_data[top_idx] * Wc[c_idx];
      top_idx += dim;
      c_idx++;
    }
  }
}

template <typename Dtype>
void VolumetricScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = top[0]->count();
  int num = top[0]->num();
  int channels = top[0]->shape(1);
  int height = top[0]->shape(2);
  int width = top[0]->shape(3);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* Wc = bottom[1]->gpu_data();
  const Dtype* Ws = bottom[2]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  ScaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, Wc, Ws, num, channels, height, width, top_data);
}

template <typename Dtype>
void VolumetricScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = top[0]->count();
  int num = top[0]->num();
  int channels = top[0]->shape(1);
  int height = top[0]->shape(2);
  int width = top[0]->shape(3);
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* Wc = bottom[1]->gpu_data();
  const Dtype* Ws = bottom[2]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* Wc_diff = bottom[1]->mutable_gpu_diff();
  Dtype* Ws_diff = bottom[2]->mutable_gpu_diff();

  ScaleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, Wc, Ws, num, channels, height, width, bottom_diff);

  ScaleBackwardSpatialWeight<Dtype><<<CAFFE_GET_BLOCKS(bottom[2]->count()), 
      CAFFE_CUDA_NUM_THREADS>>>(bottom[2]->count(), top_diff, bottom_data, Wc, 
          num, channels, height, width, Ws_diff);

  ScaleBackwardChannelWeight<Dtype><<<CAFFE_GET_BLOCKS(bottom[1]->count()), 
      CAFFE_CUDA_NUM_THREADS>>>(bottom[1]->count(), top_diff, bottom_data, Ws, 
          num, channels, height, width, Wc_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(VolumetricScaleLayer);

}  // namespace caffe
