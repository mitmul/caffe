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
PatchBasedSegmentationDataLayer<Dtype>::
~PatchBasedSegmentationDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void PatchBasedSegmentationDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // Initialize DB
  db_.reset(db::GetDB("lmdb"));
  db_->Open(
    this->layer_param_.patch_based_segmentation_data_param().source(),
    db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (
    this->layer_param_.patch_based_segmentation_data_param().rand_skip() > 0) {
    unsigned int skip =
      caffe_rng_rand() %
      this->layer_param_.patch_based_segmentation_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  top[0]->Reshape(
    this->layer_param_.patch_based_segmentation_data_param().batch_size(),
    this->layer_param_.patch_based_segmentation_data_param().data_channels(),
    this->layer_param_.patch_based_segmentation_data_param().data_height(),
    this->layer_param_.patch_based_segmentation_data_param().data_width());
  this->prefetch_data_.Reshape(
    this->layer_param_.patch_based_segmentation_data_param().batch_size(),
    this->layer_param_.patch_based_segmentation_data_param().data_channels(),
    this->layer_param_.patch_based_segmentation_data_param().data_height(),
    this->layer_param_.patch_based_segmentation_data_param().data_width());
  top[1]->Reshape(
    this->layer_param_.patch_based_segmentation_data_param().batch_size(),
    this->layer_param_.patch_based_segmentation_data_param().label_channels(),
    this->layer_param_.patch_based_segmentation_data_param().label_height(),
    this->layer_param_.patch_based_segmentation_data_param().label_width());
  this->transformed_data_.Reshape(
    1,
    this->layer_param_.patch_based_segmentation_data_param().data_channels(),
    this->layer_param_.patch_based_segmentation_data_param().data_height(),
    this->layer_param_.patch_based_segmentation_data_param().data_width());
  this->prefetch_label_.Reshape(
    this->layer_param_.patch_based_segmentation_data_param().batch_size(),
    this->layer_param_.patch_based_segmentation_data_param().label_channels(),
    this->layer_param_.patch_based_segmentation_data_param().label_height(),
    this->layer_param_.patch_based_segmentation_data_param().label_width());
}

template <typename Dtype>
void PatchBasedSegmentationDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size =
    this->layer_param_.patch_based_segmentation_data_param().batch_size();
  // const int data_channels =
  //   this->layer_param_.patch_based_segmentation_data_param().data_channels();
  const int data_height =
    this->layer_param_.patch_based_segmentation_data_param().data_height();
  const int data_width =
    this->layer_param_.patch_based_segmentation_data_param().data_width();
  const int label_channels =
    this->layer_param_.patch_based_segmentation_data_param().label_channels();
  const int label_height =
    this->layer_param_.patch_based_segmentation_data_param().label_height();
  const int label_width =
    this->layer_param_.patch_based_segmentation_data_param().label_width();
  const bool rotation =
    this->layer_param_.patch_based_segmentation_data_param().rotation();
  const bool flip =
    this->layer_param_.patch_based_segmentation_data_param().flip();
  const bool has_value =
    this->layer_param_.patch_based_segmentation_data_param().has_value();
  const bool skip_blank =
    this->layer_param_.patch_based_segmentation_data_param().skip_blank();

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
    const picojson::array img_fnames = v.get<picojson::array>();
    const string data_fname = img_fnames[0].get<string>();
    const string label_fname = img_fnames[1].get<string>();

    cv::Mat _data = cv::imread(data_fname);
    cv::Mat _label = cv::imread(label_fname, CV_LOAD_IMAGE_GRAYSCALE);

    while (1) {
      cv::Mat data = _data.clone();
      cv::Mat label = _label.clone();

      // cropping left-top point
      int _x = caffe_rng_rand() % data.cols;
      const int x = _x + data_width < data.cols ? _x : data.cols - data_width;
      int _y = caffe_rng_rand() % data.rows;
      const int y = _y + data_height < data.rows ? _y : data.rows - data_height;

      // rotation
      if (rotation) {
        const double angle = caffe_rng_rand() % 360 - 180;
        const cv::Point2f center(x + data_width / 2, y + data_height / 2);
        const cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        const cv::Scalar constant(127, 127, 127);
        cv::warpAffine(data, data, rot, cv::Size(data.cols, data.rows),
                       cv::INTER_NEAREST, cv::BORDER_CONSTANT, constant);
        cv::warpAffine(label, label, rot, cv::Size(label.cols, label.rows),
                       cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      }

      // create patch
      cv::Mat crop_data = data(cv::Rect(x, y, data_width, data_height));
      crop_data.convertTo(crop_data, CV_32F);
      cv::Mat crop_label =
        label(cv::Rect(x + data_width / 2 - label_width / 2,
                       y + data_height / 2 - label_height / 2,
                       label_width, label_height));
      crop_label.convertTo(crop_label, CV_32F);

      // skip too white patch
      if (skip_blank) {
        cv::Scalar data_sum = cv::sum(crop_data);
        if (data_sum[0] > data_width * data_height * 255 * 0.6 &&
            data_sum[1] > data_width * data_height * 255 * 0.6 &&
            data_sum[2] > data_width * data_height * 255 * 0.6)
          continue;
      }

      // must have at least 1 building or road label
      if (cv::sum(crop_label)[0] == 0 && has_value)
        continue;

      // flip
      const int flip_code = caffe_rng_rand() % 2;
      if (flip && flip_code == 1) {
        cv::flip(crop_data, crop_data, flip_code);
        cv::flip(crop_label, crop_label, flip_code);
      }

      // to multi channel label
      cv::Mat multi_ch_label(
        label_height, label_width, CV_32FC(label_channels));
      multi_ch_label =
        cv::Mat::zeros(label_height, label_width, CV_32FC(label_channels));
      for (int y = 0; y < label_height; ++y) {
        for (int x = 0; x < label_width; ++x) {
          for (int ch = 0; ch < label_channels; ++ch) {
            if (crop_label.at<float>(y, x) == ch) {
              int index = y * label_width * label_channels
                          + x * label_channels + ch;
              reinterpret_cast<float *>(multi_ch_label.data)[index] = 1.0;
            }
          }
        }
      }

      // revert to Dtype vec
      ConvertFromCVMat(crop_data,
                       top_data + this->prefetch_data_.offset(item_id));
      ConvertFromCVMat(multi_ch_label,
                       top_label + this->prefetch_label_.offset(item_id));
      break;
    }

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
cv::Mat PatchBasedSegmentationDataLayer<Dtype>::ConvertToCVMat(
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
void PatchBasedSegmentationDataLayer<Dtype>::ConvertFromCVMat(
  const cv::Mat img, Dtype *data) {
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

INSTANTIATE_CLASS(PatchBasedSegmentationDataLayer);
REGISTER_LAYER_CLASS(PatchBasedSegmentationData);

}  // namespace caffe
