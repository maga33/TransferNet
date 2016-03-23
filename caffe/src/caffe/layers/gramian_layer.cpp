#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// I assume that filter weight (bottom[1]) is num x channel

template <typename Dtype>
void GramianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(top.size(), 1);
    CHECK_EQ(bottom.size(), 1) 
        << "Number of input should be one!";
}

template <typename Dtype>
void GramianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
    int num = bottom[0]->shape(0);
    int channel = bottom[0]->shape(1);
    int width = bottom[0]->width();
    int height = bottom[0]->height();

    top[0]->Reshape(num, width*height, height, width);
    A_b.Reshape(num, width*height, height, width);
    //top[0]->Reshape(num, 1, height*width, height*width);

}

template <typename Dtype>
void GramianLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();    

    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();
 
    int M_ = height*width;
    int K_ = channels;
    int N_ = M_;

    for (int i=0; i< num; ++i){
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          bottom_data, bottom_data, (Dtype)0., top_data);
        bottom_data += M_*K_;
        top_data += M_*M_;
    }
}


template <typename Dtype>
void GramianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->cpu_data();    
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();

    int M_ = height*width;
    int K_ = channels;
    int N_ = M_;

/*
    int count = top[0]->count();
    Dtype* A_b_data = A_b.mutable_cpu_data();

    caffe_copy<Dtype>(count, top_diff, A_b_data);
    for (int i=0; i< count; ++i){
	std::cout << "CPU: copied_data " << (Dtype)A_b_data[i] << "original_data " << (Dtype)top_diff[i];
    }
    for (int i=0; i< num; ++i){
	for( int j=0; j< M_; j++){
	    A_b_data[j*M_+j] += top_diff[j*M_+j];
	    //LOG(INFO) << "CPU: copied_data " << A_b_data[j*M_+j] << "original_data " << top_diff[j*M_+j];
            //std::cout << "CPU: copied_data " << A_b_data[j*M_+j] << "original_data " << top_diff[j*M_+j] << std::endl;
	}
  	A_b_data += M_*M_;
        top_diff += M_*M_;
    }    

    A_b_data = A_b.mutable_cpu_data();

    for (int i=0; i< num; ++i){
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, M_, M_,  (Dtype)1.,
          bottom_data, A_b_data, (Dtype)0., bottom_diff);
	bottom_data += M_*K_;
        A_b_data += M_*M_;
	bottom_diff += M_*K_;
    }
*/

/*
    for (int i=0; i< num; ++i){
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, M_, M_,  (Dtype)2.,
          bottom_data, top_diff, (Dtype)0., bottom_diff);
	bottom_data += M_*K_;
        top_diff += M_*M_;
	bottom_diff += M_*K_;
    }
*/
    for (int n=0; n< num; ++n){
	for (int k=0; k<K_; ++k){
	    for (int i=0; i<M_; ++i){
		bottom_diff[k*M_+i] = 0.;
		for (int j=0; j<M_; ++j){
		    bottom_diff[k*M_+i] += bottom_data[k*M_+j] * (top_diff[i*M_+j] + top_diff[j*M_+i]);
		}
	    }
	}
	bottom_diff += K_*M_;
	bottom_data += K_*M_;
	top_diff += M_*M_;
    }

    
}


#ifdef CPU_ONLY
STUB_GPU(GramianLayer);
#endif

INSTANTIATE_CLASS(GramianLayer);
REGISTER_LAYER_CLASS(Gramian);

}  // namespace caffe
