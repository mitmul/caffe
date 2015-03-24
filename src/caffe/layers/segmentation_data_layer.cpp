#include <opencv2/opencv.hpp>

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

template <typename Dtype>
SegmentationDataLayer<Dtype>::~SegmentationDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void SegmentationDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // Initialize DB
  db_.reset(db::GetDB("lmdb"));
  db_->Open(this->layer_param_.segmentation_data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.segmentation_data_param().rand_skip()) {
    unsigned int skip =
      caffe_rng_rand() %
      this->layer_param_.segmentation_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  top[0]->Reshape(
    this->layer_param_.segmentation_data_param().batch_size(),
    this->layer_param_.segmentation_data_param().data_channels(),
    this->layer_param_.segmentation_data_param().data_height(),
    this->layer_param_.segmentation_data_param().data_width());
  this->prefetch_data_.Reshape(
    this->layer_param_.segmentation_data_param().batch_size(),
    this->layer_param_.segmentation_data_param().data_channels(),
    this->layer_param_.segmentation_data_param().data_height(),
    this->layer_param_.segmentation_data_param().data_width());
  top[1]->Reshape(
    this->layer_param_.segmentation_data_param().batch_size(),
    this->layer_param_.segmentation_data_param().label_channels(),
    this->layer_param_.segmentation_data_param().label_height(),
    this->layer_param_.segmentation_data_param().label_width());
  this->transformed_data_.Reshape(
    1,
    this->layer_param_.segmentation_data_param().data_channels(),
    this->layer_param_.segmentation_data_param().data_height(),
    this->layer_param_.segmentation_data_param().data_width());
  this->prefetch_label_.Reshape(
    this->layer_param_.segmentation_data_param().batch_size(),
    this->layer_param_.segmentation_data_param().label_channels(),
    this->layer_param_.segmentation_data_param().label_height(),
    this->layer_param_.segmentation_data_param().label_width());
}

template <typename Dtype>
void SegmentationDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size =
    this->layer_param_.segmentation_data_param().batch_size();
  const int data_channels =
    this->layer_param_.segmentation_data_param().data_channels();
  const int data_height =
    this->layer_param_.segmentation_data_param().data_height();
  const int data_width =
    this->layer_param_.segmentation_data_param().data_width();
  const int label_channels =
    this->layer_param_.segmentation_data_param().label_channels();
  const int label_height =
    this->layer_param_.segmentation_data_param().label_height();
  const int label_width =
    this->layer_param_.segmentation_data_param().label_width();


  // output of this data layer
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // Read a line
    const string line = cursor_->value();
    picojson::value v;
    string err = picojson::parse(v, line);
    if (!err.empty()) {
      LOG(FATAL) << err;
    }

    // load image
    const string img_fname =
      v.get<picojson::object>()["filename"].get<string>();
    cv::Mat img;
    img = cv::imread(img_fname);

    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

template <typename Dtype>
cv::Mat SegmentationDataLayer<Dtype>::ConvertToCVMat(
  const Dtype *data, const int &channels,
  const int &height, const int &width) {
  cv::Mat img(height, width, CV_32FC(channels));
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int index = c * height * width + h * width + w;
        float val = static_cast<float>(data[index]);
        int pos = h * width * channels + w * channels + c;
        reinterpret_cast<float *>(img.data)[pos] = val;
      }
    }
  }

  return img;
}

template <typename Dtype>
void SegmentationDataLayer<Dtype>::ConvertFromCVMat(const cv::Mat img, Dtype *data) {
  const int channels = img.channels();
  const int height = img.rows;
  const int width = img.cols;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int pos = h * width * channels + w * channels + c;
        float val = reinterpret_cast<float *>(img.data)[pos];
        const int index = c * height * width + h * width + w;
        data[index] = static_cast<Dtype>(val);
      }
    }
  }
}

INSTANTIATE_CLASS(SegmentationDataLayer);
REGISTER_LAYER_CLASS(SegmentationData);

}  // namespace caffe
