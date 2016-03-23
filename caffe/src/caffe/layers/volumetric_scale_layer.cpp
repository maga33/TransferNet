#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void VolumetricScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 3) 
      << "Second/third bottoms (scale factors) should be provided!";
}

template <typename Dtype>
void VolumetricScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) 
      << "Bottom[0] should be [Batch][Channel][Height][Width]";
  CHECK_EQ(bottom[1]->count(), bottom[0]->shape(0) * bottom[0]->shape(1)) 
      << "Bottom[1] should be Batch * Channels";
  CHECK_EQ(bottom[2]->count(), bottom[0]->shape(0) * bottom[0]->shape(2) 
          * bottom[0]->shape(3))
      << "Bottom[2] should be Batch * Height * Width";
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void VolumetricScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  const Dtype* X = bottom[0]->cpu_data();
  const Dtype* Wc = bottom[1]->cpu_data();
  const Dtype* Ws = bottom[2]->cpu_data();
  Dtype* Y = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = i * channels * height * width + c * height * width + h * width + w;
          int c_idx = i * channels + c;
          int s_idx = i * height * width + h * width + w;
          Y[idx] = X[idx] * Wc[c_idx] * Ws[s_idx];
        }
      }
    }
  }
}

template <typename Dtype>
void VolumetricScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  const Dtype* X = bottom[0]->cpu_data();
  const Dtype* Wc = bottom[1]->cpu_data();
  const Dtype* Ws = bottom[2]->cpu_data();
  const Dtype* Y_diff = top[0]->cpu_diff();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();
  Dtype* Wc_diff = bottom[1]->mutable_cpu_diff();
  Dtype* Ws_diff = bottom[2]->mutable_cpu_diff();
  caffe_set(bottom[1]->count(), (Dtype)0., Wc_diff);
  caffe_set(bottom[2]->count(), (Dtype)0., Ws_diff);
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = i * channels * height * width + c * height * width + h * width + w;
          int c_idx = i * channels + c;
          int s_idx = i * height * width + h * width + w;

          X_diff[idx] = Y_diff[idx] * Wc[c_idx] * Ws[s_idx];
          Wc_diff[c_idx] += Y_diff[idx] * X[idx] * Ws[s_idx];
          Ws_diff[s_idx] += Y_diff[idx] * X[idx] * Wc[c_idx];
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(VolumetricScaleLayer);
#endif

INSTANTIATE_CLASS(VolumetricScaleLayer);
REGISTER_LAYER_CLASS(VolumetricScale);

}  // namespace caffe
