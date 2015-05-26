#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/picojson.h"

namespace caffe {
template<typename Dtype>
BBoxDataLayer<Dtype>::~BBoxDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template<typename Dtype>
void BBoxDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype> *>& bottom,
  const vector<Blob<Dtype> *>& top) {
  batch_size_ = this->layer_param_.data_param().batch_size();
  source_     = this->layer_param_.bbox_data_param().source();
  n_category_ = this->layer_param_.bbox_data_param().n_category();
  rand_skip_  = this->layer_param_.bbox_data_param().rand_skip();
  dim_        = this->layer_param_.bbox_data_param().dim();

  // image
  top[0]->Reshape(batch_size_, dim_, 1, 1);
  this->prefetch_data_.Reshape(batch_size_, dim_, 1, 1);

  // label
  top[1]->Reshape(batch_size_, dim_, 1, 1);
  this->prefetch_label_.Reshape(batch_size_, dim_, 1, 1);

  LOG(INFO) << "output data size: "
            << top[0]->num() << ","
            << top[0]->channels() << ","
            << top[0]->height() << ","
            << top[0]->width();
  LOG(INFO) << "output label size: "
            << top[1]->num() << ","
            << top[1]->channels() << ","
            << top[1]->height() << ","
            << top[1]->width();

  json_file_ = new ifstream(source_);
}

// This function is used to create a thread that prefetches the data.
template<typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;

  batch_timer.Start();
  double   read_time  = 0;
  double   trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  Dtype *top_data  = this->prefetch_data_.mutable_cpu_data();
  Dtype *top_label = this->prefetch_label_.mutable_cpu_data();

  // per batch
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();

    // Read a line
    const string line = cursor_->value();
    picojson::value v;
    string err = picojson::parse(v, line);

    if (!err.empty()) {
      LOG(FATAL) << err;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(BBoxDataLayer);
REGISTER_LAYER_CLASS(BBoxData);
} // namespace caffe
