#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// I assume that filter weight (bottom[1]) is num x channel

template <typename Dtype>
void FilterSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(top.size(), 1);
    CHECK_EQ(bottom.size(), 2) 
        << "Second bottom (scale factor) should be provided!";
}

template <typename Dtype>
void FilterSumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
    int num = bottom[0]->shape(0);
    int width = bottom[0]->shape(2);
    int height = bottom[0]->shape(3);

    top[0]->Reshape(num, 1, width, height);
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
    Y_diff_X_.ReshapeLike(*bottom[0]);
    tmp_.Reshape(1,1, bottom[0]->shape(2), bottom[0]->shape(3));

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
void FilterSumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // net_->ForwardPrefilled();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);
    const Dtype* X = bottom[0]->cpu_data();
    const Dtype* W = bottom[1]->cpu_data();

    Dtype* Y = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), (Dtype)0., Y);
    Dtype* tmp = tmp_.mutable_cpu_data();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_cpu_scale(dim, W[j], X, tmp);
        caffe_add(dim, tmp, Y, Y);
        X += bottom[0]->count(2);
      }
      Y += dim;
      W += channels;
    }
}

template <typename Dtype>
void FilterSumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //net_->Backward();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);
    const Dtype* X = bottom[0]->cpu_data();
    const Dtype* W = bottom[1]->cpu_data();
    const Dtype* Y_diff = top[0]->cpu_diff();
    Dtype* X_diff = bottom[0]->mutable_cpu_diff();
    Dtype* W_diff = bottom[1]->mutable_cpu_diff();
    Dtype* Y_diff_X = Y_diff_X_.mutable_cpu_data();
    caffe_set(bottom[1]->count(), (Dtype)0., W_diff);

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_cpu_scale(dim, W[j], Y_diff, X_diff);
        caffe_mul(dim, Y_diff, X, Y_diff_X);
	W_diff[j] = caffe_cpu_asum(dim, Y_diff_X); // this part is wrong

        X_diff += dim;
        Y_diff_X += dim;
        X += dim;
      }
//      X += bottom[0]->count(1);
      Y_diff += dim;
      W += channels;
      W_diff += channels;
    }
}


#ifdef CPU_ONLY
STUB_GPU(FilterSumLayer);
#endif

INSTANTIATE_CLASS(FilterSumLayer);
REGISTER_LAYER_CLASS(FilterSum);

}  // namespace caffe
