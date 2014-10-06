#include <opencv2/opencv.hpp>
#include <boost/type_traits.hpp>

#include "caffe/util/io.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
LabelingDataLayer<Dtype>::~LabelingDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void LabelingDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  // Initialize DB
  LOG(INFO) << "Dataset: " << this->layer_param_.labeling_data_param().source();
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(
    mdb_env_open(mdb_env_,
                 this->layer_param_.labeling_data_param().source().c_str(),
                 MDB_RDONLY | MDB_NOTLS, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
      << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
      << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb "
            << this->layer_param_.labeling_data_param().source();
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
           MDB_SUCCESS) << "mdb_cursor_get failed";

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);

  LabelingDataParameter labeling_data_param =
    this->layer_param_.labeling_data_param();
  batch_size_ = labeling_data_param.batch_size();
  label_num_ = labeling_data_param.label_num();
  label_height_ = labeling_data_param.label_height();
  label_width_ = labeling_data_param.label_width();
  transform_ = labeling_data_param.transform();

  // data
  top[0]->Reshape(batch_size_, datum.channels(), datum.height(), datum.width());
  this->prefetch_data_.Reshape(batch_size_, datum.channels(),
                               datum.height(), datum.width());
  LOG(INFO) << "input data size: " << top[0]->num() << "," << top[0]->channels()
            << "," << top[0]->height() << "," << top[0]->width();

  // label
  top[1]->Reshape(batch_size_, 1, label_height_, label_width_);
  this->prefetch_label_.Reshape(batch_size_, 1, label_height_, label_width_);
  LOG(INFO) << "input label size: " << top[1]->num() << ","
            << top[1]->channels() << "," << top[1]->height() << ","
            << top[1]->width();

  data_channels_ = datum.channels();
  data_height_ = datum.height();
  data_width_ = datum.width();
  data_size_ = datum.channels() * datum.height() * datum.width();

  transform_param_ = this->layer_param_.transform_param();
  if (transform_param_.has_mean_file()) {
    const string &mean_file = transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
}

template <typename Dtype>
void LabelingDataLayer<Dtype>::Transform(
  Dtype *data, const int &num, const int &ch, const int &height,
  const int &width, const int &angle, const int &flipCode,
  const bool &normalize) {
  // compute channel-wise mean
  cv::Mat img(height, width, CV_32FC(ch));
  for (int c = 0; c < ch; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int index = num * height * width * ch
                    + c * height * width
                    + h * width + w;
        int pos = h * width * ch + w * ch + c;
        float val = static_cast<float>(data[index]);
        reinterpret_cast<float *>(img.data)[pos] = val;
      }
    }
  }

  for (int i = 0; i < angle / 90; ++i) {
    cv::Mat dst;
    cv::transpose(img, dst);
    cv::flip(dst, dst, 1);
    img = dst.clone();
  }

  if (flipCode > -2 && flipCode < 2) {
    cv::flip(img, img, flipCode);
  }

  // mean subtraction and stddev division is performred only for data
  if (normalize) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(img, mean, stddev);
    cv::Mat slice[ch];
    cv::split(img, slice);
    for (int c = 0; c < ch; ++c) {
      cv::subtract(slice[c], mean[c], slice[c]);
      slice[c] /= stddev[c];
    }
    cv::merge(slice, ch, img);
  }

  for (int c = 0; c < ch; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int index = num * height * width * ch
                    + c * height * width
                    + h * width + w;
        float val = reinterpret_cast<float *>(img.data)[
                      h * width * ch + w * ch + c];
        data[index] = static_cast<Dtype>(val);
      }
    }
  }
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
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                            MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);

    const string &data = datum.data();
    for (int pos = 0; pos < data_size_; ++pos) {
      int index = item_id * data_size_ + pos;
      top_data[index] = static_cast<float>(static_cast<uint8_t>(data[pos]));
    }

    const google::protobuf::RepeatedField<float> label = datum.float_data();
    const float *label_data = label.data();
    for (int pos = 0; pos < label_height_ * label_width_; ++pos) {
      int index = item_id * label_height_ * label_width_ + pos;
      top_label[index] = static_cast<float>(label_data[pos]);
    }

    // do some data augmentation
    if (transform_) {
      int angle = caffe_rng_rand() % 4 * 90;
      int flipCode = caffe_rng_rand() % 4 - 1;
      Transform(top_data, item_id, data_channels_,
                data_height_, data_width_, angle, flipCode);
      Transform(top_label, item_id, 1, label_height_, label_width_,
                angle, flipCode, false);
    }
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
                       &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
               MDB_SUCCESS);
    }
  }
}

INSTANTIATE_CLASS(LabelingDataLayer);

}  // namespace caffe