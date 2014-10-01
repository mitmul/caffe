#include <opencv2/core/core.hpp>
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
LabelingDataLayer<Dtype>::~LabelingDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void LabelingDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*> &bottom,
    vector<Blob<Dtype>*> *top) {

  // Initialize DB
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_, this->layer_param_.labeling_data_param().source().c_str(), MDB_RDONLY | MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << this->layer_param_.labeling_data_param().source();
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST), MDB_SUCCESS) << "mdb_cursor_get failed";

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  google::protobuf::RepeatedField<float> label = datum.float_data();
  label_channels_ = label.Get(0);
  label_height_ = label.Get(1);
  label_width_ = label.Get(2);
  label_data_offset_ = 3;

  LabelingDataParameter labeling_data_param = this->layer_param_.labeling_data_param();
  batch_size_ = labeling_data_param.batch_size();

  // data
  (*top)[0]->Reshape(batch_size_, datum.channels(), datum.height(), datum.width());
  this->prefetch_data_.Reshape(batch_size_, datum.channels(), datum.height(), datum.width());
  LOG(INFO) << "input data size: " << (*top)[0]->num() << "," << (*top)[0]->channels() << "," << (*top)[0]->height() << "," << (*top)[0]->width();

  // label
  (*top)[1]->Reshape(batch_size_, 1, label_height_, label_width_);
  this->prefetch_label_.Reshape(batch_size_, 1, label_height_, label_width_);
  LOG(INFO) << "input label size: " << (*top)[1]->num() << "," << (*top)[1]->channels() << "," << (*top)[1]->height() << "," << (*top)[1]->width();

  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template <typename Dtype>
void LabelingDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype *top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype *top_label = this->prefetch_label_.mutable_cpu_data();

  // datum obtains
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // get a datum
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);

    const string &data = datum.data();
    for (int pos = 0; pos < this->datum_size_; ++pos) {
      int index = item_id * this->datum_size_ + pos;
      top_data[index] = static_cast<float>(static_cast<uint8_t>(data[pos]));
    }

    const google::protobuf::RepeatedField<float> label = datum.float_data();
    for (int pos = 0; pos < label_height_ * label_width_; ++pos) {
      int index =  item_id * label_height_ * label_width_ + pos;
      top_label[index] = label.Get(label_data_offset_ + pos);
    }

    // do some data augmentation

    if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    }
  }
}

INSTANTIATE_CLASS(LabelingDataLayer);

}  // namespace caffe