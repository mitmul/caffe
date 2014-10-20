#include <opencv2/opencv.hpp>
#include <boost/type_traits.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/dataset_factory.hpp"

namespace caffe {

template <typename Dtype>
LabelingDataLayer<Dtype>::~LabelingDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the dataset resources
  dataset_->close();
}

template <typename Dtype>
void LabelingDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  // Initialize DB
  dataset_ = DatasetFactory<string, Datum>(DataParameter_DB_LMDB);
  const string &source = this->layer_param_.labeling_data_param().source();
  LOG(INFO) << "Opening dataset " << source;
  CHECK(dataset_->open(source, Dataset<string, Datum>::ReadOnly));
  iter_ = dataset_->begin();

  // Read a data point, and use it to initialize the top blob.
  CHECK(iter_ != dataset_->end());
  Datum datum = iter_->value;

  LabelingDataParameter labeling_data_param =
    this->layer_param_.labeling_data_param();
  const int batch_size = labeling_data_param.batch_size();
  const int label_height = labeling_data_param.label_height();
  const int label_width = labeling_data_param.label_width();

  // data
  top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
  this->prefetch_data_.Reshape(batch_size, datum.channels(),
                               datum.height(), datum.width());
  LOG(INFO) << "input data size: " << top[0]->num() << "," << top[0]->channels()
            << "," << top[0]->height() << "," << top[0]->width();

  // label
  top[1]->Reshape(batch_size, 1, label_height, label_width);
  this->prefetch_label_.Reshape(batch_size, 1, label_height, label_width);
  LOG(INFO) << "input label size: " << top[1]->num() << ","
            << top[1]->channels() << "," << top[1]->height() << ","
            << top[1]->width();
}

template <typename Dtype>
void LabelingDataLayer<Dtype>::Transform(
  Dtype *data, const int &num, const int &ch, const int &height,
  const int &width, const int &angle, const int &flipCode,
  const bool &transform, const bool &normalize) {
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

  if (transform) {
    for (int i = 0; i < angle / 90; ++i) {
      cv::Mat dst;
      cv::transpose(img, dst);
      cv::flip(dst, dst, 1);
      img = dst.clone();
    }

    if (flipCode > -2 && flipCode < 2) {
      cv::flip(img, img, flipCode);
    }
  }

  if (normalize) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(img, mean, stddev);
    cv::Mat *slice = new cv::Mat[ch];
    cv::split(img, slice);
    for (int c = 0; c < ch; ++c) {
      cv::subtract(slice[c], mean[c], slice[c]);
      slice[c] /= stddev[c];
    }
    cv::merge(slice, ch, img);
    delete [] slice;
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
  CHECK(this->prefetch_data_.count());
  Dtype *top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype *top_label = this->prefetch_label_.mutable_cpu_data();

  // datum obtains
  LabelingDataParameter labeling_data_param =
    this->layer_param_.labeling_data_param();
  const int batch_size = labeling_data_param.batch_size();
  const int label_num = labeling_data_param.label_num();
  const int label_height = labeling_data_param.label_height();
  const int label_width = labeling_data_param.label_width();
  const int spatial_dim = label_height * label_width;
  const bool transform = labeling_data_param.transform();
  const bool normalize = labeling_data_param.normalize();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a datum
    CHECK(iter_ != dataset_->end());
    const Datum &datum = iter_->value;
    const int channels = datum.channels();
    const int height = datum.height();
    const int width = datum.width();
    const int dim = channels * height * width;
    const string &data = datum.data();
    for (int pos = 0; pos < dim; ++pos) {
      int index = item_id * dim + pos;
      top_data[index] = static_cast<Dtype>(static_cast<uint8_t>(data[pos]));
    }

    const google::protobuf::RepeatedField<float> label = datum.float_data();
    const float *label_data = label.data();
    for (int pos = 0; pos < spatial_dim; ++pos) {
      int index = item_id * spatial_dim + pos;
      top_label[index] = static_cast<Dtype>(label_data[pos]);
    }

    // do some data augmentation
    int angle = caffe_rng_rand() % 4 * 90;
    int flipCode = caffe_rng_rand() % 4 - 1;
    // normalization(mean subtraction and stddev division)
    // should be performred only for data
    Transform(top_data, item_id, channels, height, width,
              angle, flipCode, transform, normalize);
    Transform(top_label, item_id, 1, label_height, label_width,
              angle, flipCode, transform, false);

    // go to the next iter
    ++iter_;
    if (iter_ == dataset_->end()) {
      iter_ = dataset_->begin();
    }
  }
}

INSTANTIATE_CLASS(LabelingDataLayer);
REGISTER_LAYER_CLASS(LABELING_DATA, LabelingDataLayer);

}  // namespace caffe