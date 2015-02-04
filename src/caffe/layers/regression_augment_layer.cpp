#include <opencv2/opencv.hpp>
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void RegressionAugmentLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {

  for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
    LOG(INFO) << "augment input: " << bottom[blob_id]->num() << ", "
              << bottom[blob_id]->channels() << ", "
              << bottom[blob_id]->height() << ", "
              << bottom[blob_id]->width();
  }
}

template <typename Dtype>
void RegressionAugmentLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {

  const uint32_t crop_size =
    this->layer_param_.regression_augment_param().crop_size();
  top[0]->Reshape(
    bottom[0]->num(), bottom[0]->channels(), crop_size, crop_size);
  top[1]->Reshape(
    bottom[1]->num(), bottom[1]->channels(), 1, 1);
}

template <typename Dtype>
void RegressionAugmentLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {

  CHECK_EQ(bottom.size(), top.size());
  const uint32_t crop_size =
    this->layer_param_.regression_augment_param().crop_size();

  // flip_code takes the value ranging 0, 1
  const int flip_code = caffe_rng_rand() % 2;

  // input image settings
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int label_channels = bottom[1]->channels();

  // left top point of crop region
  const int lt_x = caffe_rng_rand() % (width - crop_size);
  const int lt_y = caffe_rng_rand() % (height - crop_size);

  // foreach data in a minibatch
  for (int i = 0; i < bottom[0]->num(); ++i) {
    const Dtype *data = bottom[0]->cpu_data() + bottom[0]->offset(i);
    cv::Mat img = ConvertToCVMat(data, channels, height, width);
    Dtype *label = bottom[1]->mutable_cpu_data() + bottom[1]->offset(i);

    // randomly flipping (when flip_code == 2, it's disabled)
    if (this->layer_param_.regression_augment_param().flip()
        && flip_code == 1) {
      cv::flip(img, img, flip_code);
      for (int lc = 0; lc < label_channels; lc += 2)
        label[lc] = width - label[lc];
    }

    // crop center
    cv::Mat crop_img(crop_size, crop_size, CV_32FC(channels));
    img(cv::Rect(lt_x, lt_y, crop_size, crop_size)).copyTo(crop_img);
    for (int lc = 0; lc < label_channels; ++lc) {
      int shift = lc % 2 == 0 ? lt_x : lt_y;
      label[lc] -= shift;

      // normalize
      label[lc] = (label[lc] - crop_size / 2.0) / crop_size;
    }

    // patch-wise mean subtraction
    cv::Scalar mean, stddev;
    cv::meanStdDev(crop_img, mean, stddev);
    if (this->layer_param_.regression_augment_param().mean_normalize()) {
      cv::Mat *slice = new cv::Mat[bottom[0]->channels()];
      cv::split(crop_img, slice);
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        cv::subtract(slice[c], mean[c], slice[c]);
      }
      cv::merge(slice, bottom[0]->channels(), crop_img);
      delete [] slice;
    }

    // patch-wise stddev division
    if (this->layer_param_.regression_augment_param().stddev_normalize()) {
      cv::Mat *slice = new cv::Mat[bottom[0]->channels()];
      cv::split(crop_img, slice);
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        slice[c] /= stddev[c];
      }
      cv::merge(slice, bottom[0]->channels(), crop_img);
      delete [] slice;
    }

    // constant value subtraction
    const google::protobuf::RepeatedField<float> subs =
      this->layer_param_.regression_augment_param().subtract();
    if (subs.size() > 0) {
      CHECK_EQ(subs.size(), bottom[0]->channels());
      vector<cv::Mat> splitted;
      cv::split(crop_img, splitted);
      for (int j = 0; j < splitted.size(); ++j) {
        splitted.at(j).convertTo(splitted.at(j), CV_32F, 1.0, -subs.Get(j));
      }
      cv::merge(splitted, crop_img);
    }

    // stddev division
    const google::protobuf::RepeatedField<float> divs =
      this->layer_param_.regression_augment_param().divide();
    if (divs.size() > 0) {
      CHECK_EQ(divs.size(), bottom[0]->channels());
      vector<cv::Mat> splitted;
      cv::split(crop_img, splitted);
      for (int j = 0; j < splitted.size(); ++j) {
        splitted.at(j).convertTo(splitted.at(j), CV_32F, 1.0 / divs.Get(j));
      }
      cv::merge(splitted, crop_img);
    }

    ConvertFromCVMat(crop_img, channels, crop_size, crop_size,
                     top[0]->mutable_cpu_data() + top[0]->offset(i));
    Dtype *top_label = top[1]->mutable_cpu_data() + top[1]->offset(i);
    caffe_copy(label_channels, label, top_label);
  }
}

template <typename Dtype>
void RegressionAugmentLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom)
{
}

template <typename Dtype>
cv::Mat RegressionAugmentLayer<Dtype>::ConvertToCVMat(
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
void RegressionAugmentLayer<Dtype>::ConvertFromCVMat(
  const cv::Mat img, const int &channels, const int &height,
  const int &width, Dtype *data) {

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

INSTANTIATE_CLASS(RegressionAugmentLayer);
REGISTER_LAYER_CLASS(REGRESSION_AUGMENT, RegressionAugmentLayer);

}  // namespace caffe
