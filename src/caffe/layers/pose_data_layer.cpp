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
template<typename Dtype>
PoseDataLayer<Dtype>::~PoseDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template<typename Dtype>
void PoseDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype> *>& bottom,
  const vector<Blob<Dtype> *>& top) {
  // Initialize DB
  db_.reset(db::GetDB("lmdb"));
  db_->Open(this->layer_param_.pose_data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.pose_data_param().rand_skip() > 0) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.pose_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";

    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  const bool monochromate =
    this->layer_param_.pose_data_param().monochromate();
  int channels = this->layer_param_.pose_data_param().channels();

  if (monochromate) channels = 1;

  top[0]->Reshape(
    this->layer_param_.pose_data_param().batch_size(), channels,
    this->layer_param_.pose_data_param().height(),
    this->layer_param_.pose_data_param().width());
  this->prefetch_data_.Reshape(
    this->layer_param_.pose_data_param().batch_size(), channels,
    this->layer_param_.pose_data_param().height(),
    this->layer_param_.pose_data_param().width());
  this->transformed_data_.Reshape(
    1, channels,
    this->layer_param_.pose_data_param().height(),
    this->layer_param_.pose_data_param().width());
  top[1]->Reshape(
    this->layer_param_.pose_data_param().batch_size(),
    this->layer_param_.pose_data_param().n_joints() * 2, 1, 1);
  this->prefetch_label_.Reshape(
    this->layer_param_.pose_data_param().batch_size(),
    this->layer_param_.pose_data_param().n_joints() * 2, 1, 1);
}

template<typename Dtype>
void PoseDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;

  batch_timer.Start();
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  const bool monochromate =
    this->layer_param_.pose_data_param().monochromate();
  const int batch_size = this->layer_param_.pose_data_param().batch_size();
  int channels         = this->layer_param_.pose_data_param().channels();

  if (monochromate) {
    channels = 1;
  }

  const int  height        = this->layer_param_.pose_data_param().height();
  const int  width         = this->layer_param_.pose_data_param().width();
  const int  n_joints      = this->layer_param_.pose_data_param().n_joints();
  const bool bounding_crop =
    this->layer_param_.pose_data_param().bounding_crop();
  const bool joint_normalize =
    this->layer_param_.pose_data_param().joint_normalize();
  const bool joint_centering =
    this->layer_param_.pose_data_param().joint_centering();
  const float padding_scale_h =
    this->layer_param_.pose_data_param().padding_scale_h();
  const float padding_scale_w =
    this->layer_param_.pose_data_param().padding_scale_w();
  const int translation_size =
    this->layer_param_.pose_data_param().translation_size();
  const bool normalization =
    this->layer_param_.pose_data_param().normalization();
  const bool horizontal_flip =
    this->layer_param_.pose_data_param().horizontal_flip();
  const int rotation_angle =
    this->layer_param_.pose_data_param().rotation_angle();
  vector<int> symmetric_joint_ids;
  std::copy(
    this->layer_param_.pose_data_param().symmetric_joint_ids().begin(),
    this->layer_param_.pose_data_param().symmetric_joint_ids().end(),
    std::back_inserter(symmetric_joint_ids));
  const int left_joint_id =
    this->layer_param_.pose_data_param().left_joint_id();
  const int right_joint_id =
    this->layer_param_.pose_data_param().right_joint_id();

  // output of this data layer
  Dtype *top_data  = this->prefetch_data_.mutable_cpu_data();
  Dtype *top_label = this->prefetch_label_.mutable_cpu_data();

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

    if (monochromate) {
      img = cv::imread(img_fname, CV_LOAD_IMAGE_GRAYSCALE);
    } else {
      img = cv::imread(img_fname);
    }

    // joints
    cv::Point2f joint_center(0, 0);
    vector<cv::Point2f>   joints;
    const picojson::array joint_pos =
      v.get<picojson::object>()["joint_pos"].get<picojson::array>();
    picojson::array::const_iterator it = joint_pos.begin();

    // calc joint center location
    while (it != joint_pos.end()) {
      const picojson::array pos = it->get<picojson::array>();
      double x                  = pos[0].get<double>();
      double y                  = pos[1].get<double>();
      joints.push_back(cv::Point2f(x, y));
      joint_center += cv::Point2f(x, y);
      it++;
    }
    joint_center.x /= float(n_joints);
    joint_center.y /= float(n_joints);

    // randomly rotate
    if (rotation_angle > 0) {
      double  angle = caffe_rng_rand() % rotation_angle;
      cv::Mat rot   = cv::getRotationMatrix2D(joint_center, angle, 1.0);
      cv::Scalar constant;

      if (monochromate) {
        constant = cv::Scalar(127);
      } else {
        constant = cv::Scalar(127, 127, 127);
      }
      cv::warpAffine(img, img, rot, cv::Size(img.cols, img.rows),
                     cv::INTER_NEAREST, cv::BORDER_CONSTANT, constant);

      // rotate joints
      for (int j = 0; j < n_joints; ++j) {
        vector<cv::Point2f> pt;
        pt.push_back(joints[j]);
        cv::Mat pm = cv::Mat(pt);
        cv::transform(pm, pm, rot);
        joints[j] = pm.at<cv::Point2f>(0, 0);
      }
    }

    // crop image
    cv::Rect bounding(0, 0, width, height);
    int crop_w       = img.cols;
    int crop_h       = img.rows;
    cv::Mat crop_img = img;

    if (bounding_crop) {
      bounding = cv::boundingRect(joints);
      crop_w   = bounding.width * padding_scale_w;
      crop_h   = bounding.height * padding_scale_h;
      crop_w   = crop_w < img.cols ? crop_w : img.cols;
      crop_h   = crop_h < img.rows ? crop_h : img.rows;

      const int trans_x =
        caffe_rng_rand() % (translation_size * 2) - translation_size;
      const int trans_y =
        caffe_rng_rand() % (translation_size * 2) - translation_size;

      // adjust bounding box
      bounding.x = bounding.x - (crop_w - bounding.width) / 2 + trans_x;
      bounding.x = bounding.x >= 0 ? bounding.x : 0;
      bounding.x = (bounding.x + crop_w) <
                   img.cols ? bounding.x : img.cols - crop_w;
      bounding.width = crop_w;

      // adjust bounding box
      bounding.y = bounding.y - (crop_h - bounding.height) / 2 + trans_y;
      bounding.y = bounding.y >= 0 ? bounding.y : 0;
      bounding.y = (bounding.y + crop_h) < img.rows ?
                   bounding.y : img.rows - crop_h;
      bounding.height = crop_h;

      // crop image
      crop_img = img(bounding);

      // shift joints
      for (int j = 0; j < n_joints; ++j) {
        const int index = item_id * n_joints * 2 + j * 2;
        top_label[index + 0] = joints[j].x - bounding.x;
        top_label[index + 1] = joints[j].y - bounding.y;
      }
    }

    // convert to float mat
    crop_img.convertTo(crop_img, CV_32F);

    // resize
    cv::resize(crop_img, crop_img, cv::Size(width, height),
               0, 0, cv::INTER_NEAREST);

    // scaling joint coordinates
    for (int j = 0; j < n_joints; ++j) {
      const int index = item_id * n_joints * 2 + j * 2;
      top_label[index + 0] = float(top_label[index + 0]) / crop_w * width;
      top_label[index + 1] = float(top_label[index + 1]) / crop_h * height;
    }

    // change pixel value range
    crop_img /= 255.0;

    // global contrast normalization (image)
    if (normalization) {
      cv::Scalar mean, stddev;
      cv::meanStdDev(crop_img, mean, stddev);

      if (!monochromate) {
        cv::Mat *slice = new cv::Mat[channels];
        cv::split(crop_img, slice);

        for (int c = 0; c < channels; ++c) {
          cv::subtract(slice[c], mean[c], slice[c]);
          slice[c] /= stddev[c];
        }
        cv::merge(slice, channels, crop_img);
      } else {
        cv::subtract(crop_img, mean[0], crop_img);
        crop_img /= stddev[0];
      }
    }

    // flipping
    int flip_code = 0;

    if (horizontal_flip) {
      flip_code = caffe_rng_rand() % 2;

      if (flip_code == 1) {
        cv::flip(crop_img, crop_img, flip_code);

        // translated joints
        for (int j = 0; j < n_joints; ++j) {
          const int index = item_id * n_joints * 2 + j * 2;
          top_label[index + 0] = width - top_label[index + 0];
        }
      }
    }

    // augmented data reverting
    int offset = this->prefetch_data_.offset(item_id);
    ConvertFromCVMat(crop_img, top_data + offset);

    // joint coordinate centering
    if (joint_centering) {
      for (int j = 0; j < n_joints; ++j) {
        const int index = item_id * n_joints * 2 + j * 2;
        top_label[index + 0] -= width / 2;
        top_label[index + 1] -= height / 2;
      }
    }

    // joint coordinate normalization
    for (int j = 0; j < n_joints; ++j) {
      const int index = item_id * n_joints * 2 + j * 2;

      if (joint_normalize) {
        top_label[index + 0] /= width;
        top_label[index + 1] /= height;
      }
    }

    // receiver centric
    if ((symmetric_joint_ids.size() > 0)
        && (left_joint_id != right_joint_id)
        && (symmetric_joint_ids.size() % 2 == 0)) {
      const int l = item_id * n_joints * 2 + left_joint_id * 2;
      const int r = item_id * n_joints * 2 + right_joint_id * 2;

      if (top_label[l + 0] > top_label[r + 0]) {
        for (size_t t = 0; t < symmetric_joint_ids.size(); t = t + 2) {
          const int j       = symmetric_joint_ids.at(t);
          const int k       = symmetric_joint_ids.at(t + 1);
          const int index_j = item_id * n_joints * 2 + j * 2;
          const int index_k = item_id * n_joints * 2 + k * 2;
          std::swap(top_label[index_j + 0], top_label[index_k + 0]);
          std::swap(top_label[index_j + 1], top_label[index_k + 1]);
        }
      }
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

template<typename Dtype>
void PoseDataLayer<Dtype>::ConvertFromCVMat(const cv::Mat img, Dtype *data) {
  const int channels = img.channels();
  const int height   = img.rows;
  const int width    = img.cols;

  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int pos   = h * width * channels + w * channels + c;
        float     val   = reinterpret_cast<float *>(img.data)[pos];
        const int index = c * height * width + h * width + w;
        data[index] = static_cast<Dtype>(val);
      }
    }
  }
}

INSTANTIATE_CLASS(PoseDataLayer);
REGISTER_LAYER_CLASS(PoseData);
} // namespace caffe
