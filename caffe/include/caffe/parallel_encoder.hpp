#ifndef PARALLEL_ENCODER_HPP_
#define PARALLEL_ENCODER_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

template<typename Dtype>
class ParallelEncoder : public InternalThread {
 public:
  explicit ParallelEncoder(shared_ptr<Net<Dtype> > root_net,
      const string& param_file, Phase phase, int device_id);
  virtual ~ParallelEncoder();

  void CopyTrainedLayersFrom(const string& model_filename);
  void Sync();
 protected:
  // virtual void on_start();
  // virtual void on_gradients_ready();
  virtual void InternalThreadEntry();
  virtual void InitMemory();

  shared_ptr<Net<Dtype> > net_;
  shared_ptr<Net<Dtype> > root_net_;
  // map<string, Dtype*> gpu_ptr_;
  int gpu_;
  int root_gpu_;
};

}  // namespace caffe

#endif
