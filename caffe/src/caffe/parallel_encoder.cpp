#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel_encoder.hpp"

namespace caffe {

//
template<typename Dtype>
ParallelEncoder<Dtype>::ParallelEncoder(shared_ptr<Net<Dtype> > root_net, 
      const string& param_file, Phase phase, int device_id) : 
      root_net_(root_net), gpu_(device_id) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&root_gpu_));
  CHECK(root_gpu_ != gpu_);
  int access;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&access, gpu_, root_gpu_));
  if (access) {
    CUDA_CHECK(cudaDeviceEnablePeerAccess(root_gpu_, 0));
  } else {
    LOG(INFO) << "GPU " << gpu_ << 
        " does not have p2p access to GPU " << root_gpu_;
  }

  // Do cudaMalloc for every root blob, 
  // otherwise it could be allocated in the encoder gpu.
  const vector<shared_ptr<Blob<Dtype> > >& blobs = root_net_->blobs();
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->mutable_gpu_data();
  }

  // Create network in its own gpu memory
  CUDA_CHECK(cudaSetDevice(gpu_));
  net_.reset(new Net<Dtype>(param_file, phase));
  InitMemory();
  CUDA_CHECK(cudaSetDevice(root_gpu_));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
ParallelEncoder<Dtype>::~ParallelEncoder() {
}

template<typename Dtype>
void ParallelEncoder<Dtype>::CopyTrainedLayersFrom(const string& model_filename) {
  StopInternalThread();
  int init_gpu;
  CUDA_CHECK(cudaGetDevice(&init_gpu));
  CUDA_CHECK(cudaSetDevice(gpu_));
  net_->CopyTrainedLayersFrom(model_filename);
  CUDA_CHECK(cudaSetDevice(init_gpu));
}

// Allocate receiving buffer on parent
template<typename Dtype>
void ParallelEncoder<Dtype>::InitMemory() {
  // CUDA_CHECK(cudaSetDevice(root_gpu_)); 
  const vector<string>& names = net_->blob_names();
  const vector<int>& src_indices = net_->output_blob_indices();
  const vector<int>& dst_indices = root_net_->input_blob_indices();
  CHECK_EQ(dst_indices.size(), src_indices.size())
      << "The number of input blobs in the decoder is inconsistent with the encoder.";
  for (int i = 0; i < src_indices.size(); ++i) {
    const string& blob_name = names[src_indices[i]];
    CHECK(root_net_->has_blob(blob_name))
        << blob_name << " blob does not exist in the decoder.";
    const shared_ptr<Blob<Dtype> > dst_blob = root_net_->blob_by_name(blob_name);
    const shared_ptr<Blob<Dtype> > src_blob = net_->blob_by_name(blob_name);
    CHECK(dst_blob->shape() == src_blob->shape())
        << blob_name << " shape does not match." 
        << " Decoder: " << dst_blob->shape_string()
        << " Encoder: " << src_blob->shape_string();
    /*
    int size = src_blob->count();
    Dtype* gpu_mem_ptr = NULL;
    CUDA_CHECK(cudaMalloc(&gpu_mem_ptr, size * sizeof(Dtype)));
    gpu_ptr_[blob_name] = gpu_mem_ptr;
    */
  }

  // is this neccessary?
  // CUDA_CHECK(cudaSetDevice(gpu_)); 
  // CUDA_CHECK(cudaSetDevice(root_gpu_));
}

template<typename Dtype>
void ParallelEncoder<Dtype>::Sync() {
  StopInternalThread();
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == root_gpu_);
  CUDA_CHECK(cudaSetDevice(gpu_));

  const vector<string>& names = net_->blob_names();
  const vector<int>& src_indices = net_->output_blob_indices();
  for (int i = 0; i < src_indices.size(); ++i) {
    const string& blob_name = names[src_indices[i]];
    CHECK(root_net_->has_blob(blob_name))
        << blob_name << " blob does not exist in the decoder.";
    const shared_ptr<Blob<Dtype> > dst_blob = root_net_->blob_by_name(blob_name);
    const shared_ptr<Blob<Dtype> > src_blob = net_->blob_by_name(blob_name);
    CHECK(dst_blob->shape() == src_blob->shape())
        << blob_name << " shape does not match." 
        << " Decoder: " << dst_blob->shape_string()
        << " Encoder: " << src_blob->shape_string();

    Dtype* src = src_blob->mutable_gpu_data();
    Dtype* dst = dst_blob->mutable_gpu_data();

    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK_EQ(attributes.device, gpu_);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK_EQ(attributes.device, root_gpu_);

    CUDA_CHECK(cudaMemcpyAsync(dst, src, dst_blob->count() * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  }
  
  CUDA_CHECK(cudaSetDevice(root_gpu_));
}

template<typename Dtype>
void ParallelEncoder<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(gpu_);
  net_->ForwardPrefilled();
}

INSTANTIATE_CLASS(ParallelEncoder);

}  // namespace caffe
