#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LinearSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->shape(0);
  const Dtype* X = bottom[0]->gpu_data();
  const Dtype* W = bottom[1]->gpu_data();
  Dtype* Y = top[0]->mutable_gpu_data();
  for (int i = 0; i < num; ++i) {
    caffe_gpu_gemv(CblasNoTrans, H_, N_, (Dtype)1, X, W, (Dtype)0, Y);
    Y += top[0]->offset(1);
    X += bottom[0]->offset(1);
    W += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->shape(0);
  const Dtype* X = bottom[0]->gpu_data();
  const Dtype* W = bottom[1]->gpu_data();
  const Dtype* Y_diff = top[0]->gpu_diff();
  Dtype* X_diff = bottom[0]->mutable_gpu_diff();
  Dtype* W_diff = bottom[1]->mutable_gpu_diff();
  for (int i = 0; i < num; ++i) {
    if (propagate_down[0]) {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, H_, N_, 1,
          (Dtype)1, Y_diff, W, (Dtype)0, X_diff);
    }
    if (propagate_down[1]) {
      caffe_gpu_gemv(CblasTrans, H_, N_, (Dtype)1, X, Y_diff, 
          (Dtype)0, W_diff);
    }
    
    X += bottom[0]->offset(1);
    X_diff += bottom[0]->offset(1);
    W += bottom[1]->offset(1);
    W_diff += bottom[1]->offset(1);
    Y_diff += top[0]->offset(1);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LinearSumLayer);

}  // namespace caffe
