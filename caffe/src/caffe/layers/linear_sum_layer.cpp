#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LinearSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes());
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == 1) {
      CHECK_EQ(bottom[1]->shape(i), 1);
    }
    else {
      CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
    }
  }
  H_ = bottom[0]->shape(1);
  N_ = bottom[0]->count(2);

  LOG(INFO) << "Num of coefficient is " << N_;
  LOG(INFO) << "Output dimension is " << H_;
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape;
  shape.push_back(bottom[0]->shape(0));
  shape.push_back(H_);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->shape(0);
  const Dtype* X = bottom[0]->cpu_data();
  const Dtype* W = bottom[1]->cpu_data();
  Dtype* Y = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    caffe_cpu_gemv(CblasNoTrans, H_, N_, (Dtype)1, X, W, (Dtype)0, Y);
    Y += top[0]->offset(1);
    X += bottom[0]->offset(1);
    W += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->shape(0);
  const Dtype* X = bottom[0]->cpu_data();
  const Dtype* W = bottom[1]->cpu_data();
  const Dtype* Y_diff = top[0]->cpu_diff();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();
  Dtype* W_diff = bottom[1]->mutable_cpu_diff();
  for (int i = 0; i < num; ++i) {
    if (propagate_down[0]) {
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, H_, N_, 1,
          (Dtype)1, Y_diff, W, (Dtype)0, X_diff);
    }
    if (propagate_down[1]) {
      caffe_cpu_gemv(CblasTrans, H_, N_, (Dtype)1, X, Y_diff, 
          (Dtype)0, W_diff);
    }
    
    X += bottom[0]->offset(1);
    X_diff += bottom[0]->offset(1);
    W += bottom[1]->offset(1);
    W_diff += bottom[1]->offset(1);
    Y_diff += top[0]->offset(1);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LinearSumLayer);
#endif

INSTANTIATE_CLASS(LinearSumLayer);
REGISTER_LAYER_CLASS(LinearSum);

}  // namespace caffe
