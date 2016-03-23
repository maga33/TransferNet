#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  scale_by_param_ = this->layer_param().scale_param().has_scale_factor();
  if (scale_by_param_) {
    factor_ = this->layer_param().scale_param().scale_factor();
  }
  else {
    CHECK_EQ(top.size(), 1);
    CHECK_EQ(bottom.size(), 2) 
        << "Second bottom (scale factor) should be provided!";
    // ConstructNetwork(bottom, top);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (scale_by_param_) {
    for (int i = 0; i < top.size(); ++i) {
      top[i]->ReshapeLike(*bottom[i]);
    }
  }
  else {
    top[0]->ReshapeLike(*bottom[0]);
    CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes());
    CHECK_EQ(bottom[1]->shape(1), 1);
    for (int i = 2; i < bottom[0]->num_axes(); ++i) {
      CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
    }
    Y_diff_X_.ReshapeLike(*bottom[0]);
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
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (scale_by_param_) {
    for (int i = 0; i < top.size(); ++i) {
      caffe_cpu_scale(top[i]->count(), factor_, bottom[i]->cpu_data(), 
        top[i]->mutable_cpu_data());
    }
  }
  else {
    // net_->ForwardPrefilled();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->count(2);
    const Dtype* X = bottom[0]->cpu_data();
    const Dtype* W = bottom[1]->cpu_data();
    Dtype* Y = top[0]->mutable_cpu_data();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_mul(dim, X, W, Y);
        Y += top[0]->count(2);
        X += bottom[0]->count(2);
      }
      W += dim;
    }
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (scale_by_param_) {
    for (int i = 0; i < top.size(); ++i) {
      caffe_cpu_scale(top[i]->count(), factor_, top[i]->cpu_diff(), 
        bottom[i]->mutable_cpu_diff());
    } 
  }
  else {
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
    caffe_mul(Y_diff_X_.count(), Y_diff, X, Y_diff_X);
    caffe_set(bottom[1]->count(), (Dtype)0., W_diff);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
        caffe_mul(dim, Y_diff, W, X_diff);
        caffe_add(dim, Y_diff_X, W_diff, W_diff);
        
        X_diff += dim;
        Y_diff += dim;
        Y_diff_X += dim;
      }
      X += bottom[0]->count(1);
      W += bottom[1]->count(1);
      W_diff += dim;
    }
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::ConstructNetwork(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  NetParameter net_param;
  net_param.set_force_backward(true);

  // Input Setting
  {
    net_param.add_input("bottom");
    BlobShape* input_shape = net_param.add_input_shape();
    for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      input_shape->add_dim(bottom[0]->shape(i));
    }
  }
  {
    net_param.add_input("scale_factor");
    BlobShape* input_shape = net_param.add_input_shape();
    for (int i = 0; i < bottom[1]->num_axes(); ++i) {
      input_shape->add_dim(bottom[1]->shape(i));
    }
  }
  
  // Layer Setting
  LayerParameter* concat_layer = net_param.add_layer();
  concat_layer->set_name("concat");
  concat_layer->set_type("Concat");
  for (int i = 0; i < bottom[0]->shape(1); ++i) {
    concat_layer->add_bottom("scale_factor");
  }
  concat_layer->add_top("multiplier");

  LayerParameter* eltwise_layer = net_param.add_layer();
  eltwise_layer->set_name("multiplication");
  eltwise_layer->set_type("Eltwise");
  eltwise_layer->add_bottom("bottom");
  eltwise_layer->add_bottom("multiplier");
  eltwise_layer->add_top("top");
  eltwise_layer->mutable_eltwise_param()->
      set_operation(EltwiseParameter_EltwiseOp_PROD);

  net_.reset(new Net<Dtype>(net_param));
}

#ifdef CPU_ONLY
STUB_GPU(ScaleLayer);
#endif

INSTANTIATE_CLASS(ScaleLayer);
REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
