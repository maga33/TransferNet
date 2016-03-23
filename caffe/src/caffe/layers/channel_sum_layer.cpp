#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// I assume that filter weight (bottom[1]) is num x channel

template <typename Dtype>
void ChannelSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(top.size(), 1);
    CHECK_EQ(bottom.size(), 2) 
        << "Second bottom (scale factor) should be provided!";
}

template <typename Dtype>
void ChannelSumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
    int num = bottom[0]->shape(0);
    int width = bottom[0]->width();
    int height = bottom[0]->height();
    int channels = bottom[0]->shape(1);

    //top[0]->Reshape(num, channels, 1, 1);
    top[0]->ReshapeLike(*bottom[1]);
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));

    /*
    net_->blob_by_name("bottom")->ReshapeLike(*bottom[0]);
    net_->blob_by_name("scale_factor")->ReshapeLike(*bottom[1]);
    net_->Reshape();
    top[0]->ReshapeLike(*net_->blob_by_name("top").get());

    net_->blob_by_name("bottom")->ShareData(*bottom[0]);
    net_->blob_by_name("bottom")->ShareDiff(*bottom[0]);
    net_->blob_by_name("scale_factor")->ShareData(*bottom[1]);
    net_->blob_by_name("scale_factor")->ShareDiff(*bottom[1]);
    net_->blob_by_name("top")->ShareData(*top[0]);
    net_->blob_by_name("top")->ShareDiff(*top[0]);
    */
}

template <typename Dtype>
void ChannelSumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


    // net_->ForwardPrefilled();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);
    const Dtype* X = bottom[0]->cpu_data();
    const Dtype* W = bottom[1]->cpu_data();

    Dtype* Y = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), (Dtype)0., Y);

    for (int i = 0; i < num; ++i) {
      for (int c = 0; c < channels; ++c){
        for (int j = 0; j < dim; ++j) {
          Y[c] += W[c]*X[j];
        }
        X += dim;
      }
      Y += channels;
      W += channels;
    }

}

template <typename Dtype>
void ChannelSumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);

    const Dtype* X = bottom[0]->cpu_data();
    const Dtype* W = bottom[1]->cpu_data();
    const Dtype* Y_diff = top[0]->cpu_diff();

    Dtype* X_diff = bottom[0]->mutable_cpu_diff();
    Dtype* W_diff = bottom[1]->mutable_cpu_diff();

    caffe_set(bottom[1]->count(), (Dtype)0., W_diff);

    for (int i = 0; i < num; ++i) {
      for (int c = 0; c < channels; ++c) {
        for (int j = 0; j < dim; ++j){
	  X_diff[j] = W[c]*Y_diff[c];  // update data diff 
	  W_diff[c] += Y_diff[c]*X[j]; // update weight diff 
        }
        X_diff += dim;
        X += dim;
      }
      W += channels;
      W_diff += channels;
      Y_diff += channels;
    }

}


#ifdef CPU_ONLY
STUB_GPU(ChannelSumLayer);
#endif

INSTANTIATE_CLASS(ChannelSumLayer);
REGISTER_LAYER_CLASS(ChannelSum);

}  // namespace caffe
