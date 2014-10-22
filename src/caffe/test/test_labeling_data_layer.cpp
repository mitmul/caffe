#include <opencv2/opencv.hpp>

#include "lmdb.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/dataset_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LabelingDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LabelingDataLayerTest()
    : blob_top_data_(new Blob<Dtype>()),
      blob_top_label_(new Blob<Dtype>()),
      seed_(25314) {
    Caffe::set_random_seed(seed_);
  }

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the LMDB with data
  void FillLMDB_multi() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db_multi";

    LOG(INFO) << "Using temporary lmdb " << *filename_;
    CHECK_EQ(mkdir(filename_->c_str(), 0744), 0)
        << "mkdir " << filename_ << "failed";

    MDB_env *env;
    MDB_dbi dbi;
    MDB_val mdbkey, mdbdata;
    MDB_txn *txn;
    CHECK_EQ(mdb_env_create(&env), MDB_SUCCESS)
        << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(env, 1099511627776), MDB_SUCCESS)
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(env, filename_->c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(env, NULL, 0, &txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(txn, NULL, 0, &dbi), MDB_SUCCESS)
        << "mdb_open failed";

    const int num = 5;
    const int data_ch = 3;
    const int data_h = 2;
    const int data_w = 3;
    const int spatial_dim = data_h * data_w;
    const int dim = data_ch * data_h * data_w;
    for (int i = 0; i < num; ++i) {
      Datum datum;
      datum.set_label(0);
      datum.set_channels(data_ch);
      datum.set_height(data_h);
      datum.set_width(data_w);

      // fill data
      std::string *data = datum.mutable_data();
      for (int c = 0; c < data_ch; ++c) {
        for (int y = 0; y < data_h; ++y) {
          for (int x = 0; x < data_w; ++x) {
            const int index = i * dim + c * spatial_dim + y * data_w + x;
            data->push_back(static_cast<uint8_t>(index));
          }
        }
      }
      // fill label
      google::protobuf::RepeatedField<float> *label =
        datum.mutable_float_data();
      for (int y = 0; y < data_h; ++y) {
        for (int x = 0; x < data_w; ++x) {
          label->Add(static_cast<float>(y * data_w + x));
        }
      }
      stringstream ss;
      ss << i;

      string value;
      datum.SerializeToString(&value);
      mdbdata.mv_size = value.size();
      mdbdata.mv_data = reinterpret_cast<void *>(&value[0]);
      string keystr = ss.str();
      mdbkey.mv_size = keystr.size();
      mdbkey.mv_data = reinterpret_cast<void *>(&keystr[0]);
      CHECK_EQ(mdb_put(txn, dbi, &mdbkey, &mdbdata, 0), MDB_SUCCESS)
          << "mdb_put failed";
    }
    CHECK_EQ(mdb_txn_commit(txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(env, dbi);
    mdb_env_close(env);
  }

  // Fill the LMDB with data
  void FillLMDB_binary() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db_binary";

    LOG(INFO) << "Using temporary lmdb " << *filename_;
    CHECK_EQ(mkdir(filename_->c_str(), 0744), 0)
        << "mkdir " << filename_ << "failed";

    MDB_env *env;
    MDB_dbi dbi;
    MDB_val mdbkey, mdbdata;
    MDB_txn *txn;
    CHECK_EQ(mdb_env_create(&env), MDB_SUCCESS)
        << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(env, 1099511627776), MDB_SUCCESS)
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(env, filename_->c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(env, NULL, 0, &txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(txn, NULL, 0, &dbi), MDB_SUCCESS)
        << "mdb_open failed";

    const int num = 5;
    const int data_ch = 3;
    const int data_h = 2;
    const int data_w = 3;
    const int spatial_dim = data_h * data_w;
    const int dim = data_ch * data_h * data_w;
    for (int i = 0; i < num; ++i) {
      // fill data
      Datum datum;
      datum.set_label(0);
      datum.set_channels(data_ch);
      datum.set_height(data_h);
      datum.set_width(data_w);
      std::string *data = datum.mutable_data();
      for (int c = 0; c < data_ch; ++c) {
        for (int y = 0; y < data_h; ++y) {
          for (int x = 0; x < data_w; ++x) {
            const int index = i * dim + c * spatial_dim + y * data_w + x;
            data->push_back(static_cast<uint8_t>(index));
          }
        }
      }
      // fill label
      google::protobuf::RepeatedField<float> *label =
        datum.mutable_float_data();
      for (int y = 0; y < data_h; ++y) {
        for (int x = 0; x < data_w; ++x) {
          label->Add(static_cast<float>((y * data_w + x) % 2));
        }
      }
      stringstream ss;
      ss << i;

      string value;
      datum.SerializeToString(&value);
      mdbdata.mv_size = value.size();
      mdbdata.mv_data = reinterpret_cast<void *>(&value[0]);
      string keystr = ss.str();
      mdbkey.mv_size = keystr.size();
      mdbkey.mv_data = reinterpret_cast<void *>(&keystr[0]);
      CHECK_EQ(mdb_put(txn, dbi, &mdbkey, &mdbdata, 0), MDB_SUCCESS)
          << "mdb_put failed";
    }
    CHECK_EQ(mdb_txn_commit(txn), MDB_SUCCESS)
        << "mdb_txn_commit failed";
    mdb_close(env, dbi);
    mdb_env_close(env);
  }

  virtual ~LabelingDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  shared_ptr<string> filename_;
  Blob<Dtype> *const blob_top_data_;
  Blob<Dtype> *const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(LabelingDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelingDataLayerTest, TestRead_multi) {
  typedef typename TypeParam::Dtype Dtype;
  this->FillLMDB_multi();
  const int num = 5;
  const int data_ch = 3;
  const int data_h = 2;
  const int data_w = 3;
  const int label_classes = 6;

  LayerParameter param;
  LabelingDataParameter *labeling_data_param =
    param.mutable_labeling_data_param();
  labeling_data_param->set_batch_size(num);
  labeling_data_param->set_source(this->filename_->c_str());
  labeling_data_param->set_label_classes(label_classes);
  labeling_data_param->set_label_height(data_h);
  labeling_data_param->set_label_width(data_w);
  labeling_data_param->set_transform(false);
  labeling_data_param->set_normalize(false);

  LabelingDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_data_->num(), num);
  EXPECT_EQ(this->blob_top_data_->channels(), data_ch);
  EXPECT_EQ(this->blob_top_data_->height(), data_h);
  EXPECT_EQ(this->blob_top_data_->width(), data_w);

  EXPECT_EQ(this->blob_top_label_->num(), num);
  EXPECT_EQ(this->blob_top_label_->channels(), label_classes);
  EXPECT_EQ(this->blob_top_label_->height(), data_h);
  EXPECT_EQ(this->blob_top_label_->width(), data_w);

  const int spatial_dim = data_h * data_w;
  const int dim = this->blob_top_data_->count() / num;
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < data_ch; ++c) {
      for (int y = 0; y < data_h; ++y) {
        for (int x = 0; x < data_w; ++x) {
          const int index = i * dim + c * spatial_dim + y * data_w + x;
          EXPECT_EQ(index, this->blob_top_data_->cpu_data()[index]);
        }
      }
    }
  }

  const int label_dim = this->blob_top_label_->count() / num;
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < label_classes; ++c) {
      for (int y = 0; y < data_h; ++y) {
        for (int x = 0; x < data_w; ++x) {
          const int index = i * label_dim + c * spatial_dim + y * data_w + x;
          const int label = this->blob_top_label_->cpu_data()[index];
          if (c == y * data_w + x) {
            EXPECT_EQ(label, 1);
          } else {
            EXPECT_EQ(label, 0);
          }
        }
      }
    }
  }
}

TYPED_TEST(LabelingDataLayerTest, TestRead_binary) {
  typedef typename TypeParam::Dtype Dtype;
  this->FillLMDB_binary();
  const int num = 5;
  const int data_ch = 3;
  const int data_h = 2;
  const int data_w = 3;
  const int label_classes = 1;

  LOG(INFO) << this->filename_->c_str();
  LayerParameter param;
  LabelingDataParameter *labeling_data_param =
    param.mutable_labeling_data_param();
  labeling_data_param->set_batch_size(num);
  labeling_data_param->set_source(this->filename_->c_str());
  labeling_data_param->set_label_classes(label_classes);
  labeling_data_param->set_label_height(data_h);
  labeling_data_param->set_label_width(data_w);
  labeling_data_param->set_transform(false);
  labeling_data_param->set_normalize(false);

  LabelingDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_data_->num(), num);
  EXPECT_EQ(this->blob_top_data_->channels(), data_ch);
  EXPECT_EQ(this->blob_top_data_->height(), data_h);
  EXPECT_EQ(this->blob_top_data_->width(), data_w);

  EXPECT_EQ(this->blob_top_label_->num(), num);
  EXPECT_EQ(this->blob_top_label_->channels(), label_classes);
  EXPECT_EQ(this->blob_top_label_->height(), data_h);
  EXPECT_EQ(this->blob_top_label_->width(), data_w);

  const int spatial_dim = data_h * data_w;
  const int dim = this->blob_top_data_->count() / num;
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < data_ch; ++c) {
      for (int y = 0; y < data_h; ++y) {
        for (int x = 0; x < data_w; ++x) {
          const int index = i * dim + c * spatial_dim + y * data_w + x;
          EXPECT_EQ(index, this->blob_top_data_->cpu_data()[index]);
        }
      }
    }
  }

  const int label_dim = this->blob_top_label_->count() / num;
  EXPECT_EQ(label_dim, data_h * data_w);
  for (int i = 0; i < num; ++i) {
    for (int y = 0; y < data_h; ++y) {
      for (int x = 0; x < data_w; ++x) {
        const int index = i * label_dim + y * data_w + x;
        const int label = this->blob_top_label_->cpu_data()[index];
        EXPECT_EQ(label, (y * data_w + x) % 2);
      }
    }
  }
}

TYPED_TEST(LabelingDataLayerTest, TestLMDB) {
  typedef typename TypeParam::Dtype Dtype;

  string out_dir = CMAKE_SOURCE_DIR "caffe/test/test_data" CMAKE_EXT;
  string db_file =
    CMAKE_SOURCE_DIR "caffe/test/test_data/sample_labeling_data.lmdb" CMAKE_EXT;
  if (fopen(db_file.c_str(), "r") != NULL) {
    const unsigned int seed = (unsigned) time(NULL);
    Caffe::set_random_seed(seed);

    const int batch_size = 100;
    LayerParameter param;
    LabelingDataParameter *labeling_data_param =
      param.mutable_labeling_data_param();
    labeling_data_param->set_batch_size(batch_size);
    labeling_data_param->set_source(db_file.c_str());
    labeling_data_param->set_label_classes(1);
    labeling_data_param->set_label_height(16);
    labeling_data_param->set_label_width(16);
    labeling_data_param->set_transform(true);
    labeling_data_param->set_normalize(true);

    LabelingDataLayer<Dtype> layer(param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_data_->num(), batch_size);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    EXPECT_EQ(this->blob_top_data_->height(), 64);
    EXPECT_EQ(this->blob_top_data_->width(), 64);

    EXPECT_EQ(this->blob_top_label_->num(), batch_size);
    EXPECT_EQ(this->blob_top_label_->channels(), 1);
    EXPECT_EQ(this->blob_top_label_->height(), 16);
    EXPECT_EQ(this->blob_top_label_->width(), 16);

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const int num = this->blob_top_data_->num();
    const int channels = this->blob_top_data_->channels();
    const int height = this->blob_top_data_->height();
    const int width = this->blob_top_data_->width();
    const int dim = channels * height * width;
    const int spatial_dim = width * height;
    for (int i = 0; i < num; ++i) {
      cv::Mat img(height, width, CV_32FC(channels));
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            float pix = this->blob_top_data_->cpu_data()[
                          i * dim + c * spatial_dim + y * width + x];
            ((float *)img.data)[y * width * channels + x * channels + c] = pix;
          }
        }
      }
      cv::Scalar mean, stddev;
      cv::meanStdDev(img, mean, stddev);
      for (int c = 0; c < channels; ++c) {
        EXPECT_NEAR(mean[c], 0, 1e-5);
        EXPECT_NEAR(stddev[c], 1, 1e-5);
      }
      cv::Mat dst;
      cv::normalize(img, dst, 0, 255, cv::NORM_MINMAX);
      dst.convertTo(dst, CV_8U);
      stringstream ss;
      ss << out_dir << "/test_data_" << i << ".png";
      cv::imwrite(ss.str(), dst);
    }

    const int lnum = this->blob_top_label_->num();
    const int lchannels = this->blob_top_label_->channels();
    const int lheight = this->blob_top_label_->height();
    const int lwidth = this->blob_top_label_->width();
    const int ldim = lchannels * lheight * lwidth;
    for (int i = 0; i < lnum; ++i) {
      cv::Mat label(lheight, lwidth, CV_32FC(lchannels));
      for (int y = 0; y < lheight; ++y) {
        for (int x = 0; x < lwidth; ++x) {
          float pix = this->blob_top_label_->cpu_data()[
                        i * ldim + y * lwidth + x];
          ((float *)label.data)[y * lwidth + x] = pix;
        }
      }
      cv::Mat dst;
      cv::normalize(label, dst, 0, 255, cv::NORM_MINMAX);
      dst.convertTo(dst, CV_8U);
      stringstream ss;
      ss << out_dir << "/test_label_" << i << ".png";
      cv::imwrite(ss.str(), dst);
    }
  }
}

}  // namespace caffe