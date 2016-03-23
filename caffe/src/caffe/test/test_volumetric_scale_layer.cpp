#include <cmath>
#include <cstdlib>
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

template <typename TypeParam>
class VolumetricScaleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  VolumetricScaleLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_weight_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_weight2_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_weight_);
    filler.Fill(this->blob_weight2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_weight_);
    blob_bottom_vec_.push_back(blob_weight2_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~VolumetricScaleLayerTest() {
    delete blob_bottom_;
    delete blob_weight_;
    delete blob_weight2_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_weight_;
  Blob<Dtype>* const blob_weight2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(VolumetricScaleLayerTest, TestDtypesAndDevices);

TYPED_TEST(VolumetricScaleLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  VolumetricScaleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->cpu_data()[this->blob_bottom_->offset(i, j, h, w)]
              * this->blob_weight_->cpu_data()[this->blob_weight_->offset(i, j)]
              * this->blob_weight2_->cpu_data()[this->blob_weight2_->offset(i, 0, h, w)], 
              this->blob_top_->cpu_data()[this->blob_top_->offset(i, j, h, w)]);
        }
      }
    }
  }
}

TYPED_TEST(VolumetricScaleLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  VolumetricScaleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
