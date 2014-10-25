#include <opencv2/opencv.hpp>
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void AugmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
                                  const vector<Blob<Dtype>*> &top) {
  const int data_size =
    this->layer_param_.augment_param().data_crop_size();
  const int label_size =
    this->layer_param_.augment_param().label_crop_size();

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                  data_size, data_size);
  top[1]->Reshape(bottom[1]->num(), bottom[1]->channels(),
                  label_size, label_size);
}

template <typename Dtype>
void AugmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                                      const vector<Blob<Dtype>*> &top) {
  const int data_channels = bottom[0]->channels();
  const int data_height = bottom[0]->height();
  const int data_width = bottom[0]->width();
  CHECK_EQ(data_height, data_width);

  const int label_channels = bottom[1]->channels();
  const int label_height = bottom[1]->height();
  const int label_width = bottom[1]->width();
  CHECK_EQ(label_height, label_width);

  for (int i = 0; i < bottom[0]->num(); ++i) {
    const Dtype *data = bottom[0]->cpu_data() + bottom[0]->offset(i);
    cv::Mat data_img = ConvertToCVMat(data, data_channels,
                                      data_height, data_width);
    const Dtype *label = bottom[1]->cpu_data() + bottom[1]->offset(i);
    cv::Mat label_img = ConvertToCVMat(label, label_channels,
                                       label_height, label_width);

    // randomly rotate
    if (this->layer_param_.augment_param().rotate()) {
      const double angle = static_cast<double>(caffe_rng_rand() % 360);
      cv::Point2f data_pt(data_width / 2.0, data_height / 2.0);
      cv::Mat data_rot = cv::getRotationMatrix2D(data_pt, angle, 1.0);
      cv::warpAffine(data_img, data_img, data_rot,
                     cv::Size(data_width, data_height));
      cv::Point2f label_pt(label_width / 2.0, label_height / 2.0);
      cv::Mat label_rot = cv::getRotationMatrix2D(label_pt, angle, 1.0);
      cv::warpAffine(label_img, label_img, label_rot,
                     cv::Size(label_width, label_height));
    }

    // crop center
    const int data_size =
      this->layer_param_.augment_param().data_crop_size();
    const int label_size =
      this->layer_param_.augment_param().label_crop_size();
    cv::Mat data_patch(data_size, data_size, CV_64FC(data_channels));
    data_img(cv::Rect(data_width / 2 - data_size / 2,
                      data_height / 2 - data_size / 2,
                      data_size, data_size)).copyTo(data_patch);
    cv::Mat label_patch(label_size, label_size, CV_64FC(label_channels));
    label_img(cv::Rect(label_width / 2 - label_size / 2,
                       label_height / 2 - label_size / 2,
                       label_size, label_size)).copyTo(label_patch);

    // mean subtraction
    const google::protobuf::RepeatedField<float> mean =
      this->layer_param_.augment_param().mean();
    if (mean.size() > 0) {
      CHECK_EQ(mean.size(), data_channels);
      vector<cv::Mat> splitted;
      cv::split(data_patch, splitted);
      for (int j = 0; j < splitted.size(); ++j) {
        splitted.at(j).convertTo(splitted.at(j), CV_64F, 1.0, -mean.Get(j));
      }
      cv::merge(splitted, data_patch);
    }

    // stddev division
    const google::protobuf::RepeatedField<float> stddev =
      this->layer_param_.augment_param().stddev();
    if (stddev.size() > 0) {
      CHECK_EQ(stddev.size(), data_channels);
      vector<cv::Mat> splitted;
      cv::split(data_patch, splitted);
      for (int j = 0; j < splitted.size(); ++j) {
        splitted.at(j).convertTo(splitted.at(j), CV_64F, 1.0 / stddev.Get(j));
      }
      cv::merge(splitted, data_patch);
    }

    // revert into blob
    ConvertFromCVMat(data_patch,
                     top[0]->mutable_cpu_data() + top[0]->offset(i));
    ConvertFromCVMat(label_patch,
                     top[1]->mutable_cpu_data() + top[1]->offset(i));
  }
}

template <typename Dtype>
cv::Mat AugmentLayer<Dtype>::ConvertToCVMat(
  const Dtype *data, const int &channels,
  const int &height, const int &width) {
  cv::Mat img(height, width, CV_64FC(channels));
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int index = c * height * width + h * width + w;
        double val = static_cast<double>(data[index]);
        int pos = h * width * channels + w * channels + c;
        reinterpret_cast<double *>(img.data)[pos] = val;
      }
    }
  }

  return img;
}

template <typename Dtype>
void AugmentLayer<Dtype>::ConvertFromCVMat(const cv::Mat img, Dtype *data) {
  const int channels = img.channels();
  const int height = img.rows;
  const int width = img.cols;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int pos = h * width * channels + w * channels + c;
        double val = reinterpret_cast<double *>(img.data)[pos];
        const int index = c * height * width + h * width + w;
        data[index] = static_cast<Dtype>(val);
      }
    }
  }
}

INSTANTIATE_CLASS(AugmentLayer);
REGISTER_LAYER_CLASS(AUGMENT, AugmentLayer);

}  // namespace caffe
