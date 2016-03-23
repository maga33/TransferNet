#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class FilterSumLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  FilterSumLayerTest()
      : blob_bottom_x_(new Blob<Dtype>(2, 3, 5, 5)),
        blob_bottom_w_(new Blob<Dtype>(2, 3, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_x_);
    filler.Fill(this->blob_bottom_w_);
    blob_bottom_vec_.push_back(blob_bottom_x_);
    blob_bottom_vec_.push_back(blob_bottom_w_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~FilterSumLayerTest() {
    delete blob_bottom_x_;
    delete blob_bottom_w_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_x_;
  Blob<Dtype>* const blob_bottom_w_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FilterSumLayerTest, TestDtypesAndDevices);

TYPED_TEST(FilterSumLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<FilterSumLayer<Dtype> > layer(
      new FilterSumLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 1);
}

TYPED_TEST(FilterSumLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FilterSumLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
