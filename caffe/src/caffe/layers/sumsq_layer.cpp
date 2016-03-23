#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// I assume that filter weight (bottom[1]) is num x channel

template <typename Dtype>
void SumSqLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(top.size(), 1);
    CHECK_EQ(bottom.size(), 1) 
        << "Second bottom (scale factor) should be provided!";
}

template <typename Dtype>
void SumSqLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
    int num = bottom[0]->shape(0);
    int width = bottom[0]->shape(2);
    int height = bottom[0]->shape(3);

    top[0]->Reshape(num, 1, width, height);

}

template <typename Dtype>
void SumSqLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void SumSqLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(SumSqLayer);
#endif

INSTANTIATE_CLASS(SumSqLayer);
REGISTER_LAYER_CLASS(SumSq);

}  // namespace caffe
