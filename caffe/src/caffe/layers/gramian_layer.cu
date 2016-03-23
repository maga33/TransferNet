#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
__global__ void GramianTopDiff(const int nthreads, const Dtype* top_diff, const Dtype* bottom_data, 
    const int D, const int M, Dtype* bottom_diff) {
  // dim = M*M;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / D / M;
    int k = (index / M) % D;
    int i = index % M;

    int bottom_diff_idx = n*D*M + k*M + i;
    int bottom_data_idx = n*D*M + k*M;
    int top_diff_idx = n*M*M;
    bottom_diff[bottom_diff_idx]=0;
    for (int j=0; j< M; ++j){
	bottom_diff[bottom_diff_idx] += bottom_data[bottom_data_idx+j]*(top_diff[top_diff_idx+i*M+j] + top_diff[top_diff_idx+j*M+i]);
    }
    
  }
}



template <typename Dtype>
void GramianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//    Forward_cpu(bottom, top);

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();    

    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();
 

    int M_ = height*width;
    int K_ = channels;
    int N_ = M_;

    for (int i=0; i< num; ++i){
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          bottom_data, bottom_data, (Dtype)0., top_data);
        bottom_data += M_*K_;
        top_data += M_*M_;
    }

}


template <typename Dtype>
void GramianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//    Backward_cpu(top, propagate_down, bottom);
    const Dtype* bottom_data = bottom[0]->gpu_data();    
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int channels = bottom[0]->shape(1);
    int height = bottom[0]->height();
    int width = bottom[0]->width();

    int M_ = height*width;
    int K_ = channels;
    int N_ = M_;

    int count = bottom[0]->count();

    GramianTopDiff<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data,  K_, M_, bottom_diff);

/*
    int count = top[0]->count();
    Dtype* top_diff_A = A_b.mutable_gpu_data();

    GramianTopDiff_A<Dtype><<<CAFFE_GET_BLOCKS(num*M_*M_), CAFFE_CUDA_NUM_THREADS>>>(
        num*M_*M_, top_diff, num, M_*M_, M_, top_diff_A);


    
    caffe_gpu_set<Dtype>(count, (Dtype)0., A_b_data);
    caffe_gpu_add<Dtype>(count, top_diff, A_b_data, A_b_data);
    for (int i=0; i< num; ++i){
	for( int j=0; j< M_; j++){
	    A_b_data[j*M_+j] += top_diff[j*M_+j];
	    LOG(INFO) << "copied_data " << A_b_data[j*M_+j] << "original_data " << top_diff[j*M_+j];
	    std::cout << "copied_data " << A_b_data[j*M_+j] << "original_data " << top_diff[j*M_+j] << std::endl;
	}
  	A_b_data += M_*M_;
        top_diff += M_*M_;
    }    
    
    A_b_data = A_b.mutable_gpu_data();
    
    
    for (int i=0; i< num; ++i){
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, M_, M_,  (Dtype)1.,
          bottom_data, top_diff_A, (Dtype)0., bottom_diff);
	bottom_data += M_*K_;
        top_diff_A += M_*M_;
	bottom_diff += M_*K_;
    }
    */
/*
    for (int i=0; i< num; ++i){
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, M_, M_,  (Dtype)2.,
          bottom_data, top_diff, (Dtype)0., bottom_diff);
	bottom_data += M_*K_;
        top_diff += M_*M_;
	bottom_diff += M_*K_;
    }
*/
}
INSTANTIATE_LAYER_GPU_FUNCS(GramianLayer);

}
