//
// Created by HenryZeng on 2020/10/30.
//

#ifndef MASK_CHECK_H
#define MASK_CHECK_H

#include <string>
#include <vector>
#include <arm_neon.h>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "paddle_api.h"

using namespace paddle::lite_api; // NOLINT

struct Object
{
  int batch_id;
  cv::Rect rec;
  int class_id;
  float prob;
};

int64_t ShapeProduction(const shape_t &shape)
{
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float *din,
                     float *dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale)
{
  if (mean.size() != 3 || scale.size() != 3)
  {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4)
  {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++)
  {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

cv::Mat crop_img(const cv::Mat &img,
                 cv::Rect rec,
                 int res_width,
                 int res_height)
{
  float xmin = rec.x;
  float ymin = rec.y;
  float w = rec.width;
  float h = rec.height;
  float center_x = xmin + w / 2;
  float center_y = ymin + h / 2;
  cv::Point2f center(center_x, center_y);
  float max_wh = std::max(w / 2, h / 2);
  float scale = res_width / (2 * max_wh * 1.5);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, 0.f, scale);
  rot_mat.at<double>(0, 2) =
      rot_mat.at<double>(0, 2) - (center_x - res_width / 2.0);
  rot_mat.at<double>(1, 2) =
      rot_mat.at<double>(1, 2) - (center_y - res_width / 2.0);
  cv::Mat affine_img;
  cv::warpAffine(img, affine_img, rot_mat, cv::Size(res_width, res_height));
  return affine_img;
}

void pre_process(const cv::Mat &img,
                 int width,
                 int height,
                 const std::vector<float> &mean,
                 const std::vector<float> &scale,
                 float *data,
                 bool is_scale = false)
{
  cv::Mat resized_img;
  if (img.cols != width || img.rows != height)
  {
    cv::resize(
        img, resized_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
  }
  else
  {
    resized_img = img;
  }
  cv::Mat imgf;
  float scale_factor = is_scale ? 1.f / 256 : 1.f;
  resized_img.convertTo(imgf, CV_32FC3, scale_factor);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  neon_mean_scale(dimg, data, width * height, mean, scale);
}

class Check
{
private:
  cv::Mat img;
  float shrink = 0.4;
  int width;
  int height;
  int s_width;
  int s_height;

  std::shared_ptr<PaddlePredictor> predictor_First;
  std::shared_ptr<PaddlePredictor> predictor_Second;

  std::vector<Object> detect_result;

  std::string json, x, y, w, h, p;

public:
  void LoadModel(std::string det_model_file, std::string class_model_file);

  void FirstStep();

  void SecondStep();

  const char *Sync(cv::Mat img_, int cols, int rows);
};

void Check::LoadModel(std::string det_model_file, std::string class_model_file)
{
  MobileConfig config;
  config.set_model_from_file(det_model_file);

  // Create Predictor For Detction Model
  predictor_First = CreatePaddlePredictor<MobileConfig>(config);

  config.set_model_from_file(class_model_file);

  // Create Predictor For Classification Model
  predictor_Second = CreatePaddlePredictor<MobileConfig>(config);
}

void Check::FirstStep()
{
  detect_result.clear();

  std::unique_ptr<Tensor> input_tensor0(std::move(predictor_First->GetInput(0)));
  input_tensor0->Resize({1, 3, s_height, s_width});
  auto *data = input_tensor0->mutable_data<float>();

  // Do PreProcess
  std::vector<float> detect_mean = {104.f, 117.f, 123.f};
  std::vector<float> detect_scale = {0.007843, 0.007843, 0.007843};
  pre_process(img, s_width, s_height, detect_mean, detect_scale, data, false);

  // Detection Model Run
  predictor_First->Run();

  // Get Output Tensor
  std::unique_ptr<const Tensor> output_tensor0(
      std::move(predictor_First->GetOutput(0)));
  auto *outptr = output_tensor0->data<float>();
  auto shape_out = output_tensor0->shape();
  int64_t out_len = ShapeProduction(shape_out);
  //    std::cout << "Detecting face succeed." << std::endl;

  // Filter Out Detection Box
  float detect_threshold = 0.7;
  // =======================================================

  for (int i = 0; i < out_len / 6; ++i)
  {
    if (outptr[1] >= detect_threshold)
    {
      Object obj;
      int xmin = static_cast<int>(width * outptr[2]);
      int ymin = static_cast<int>(height * outptr[3]);
      int xmax = static_cast<int>(width * outptr[4]);
      int ymax = static_cast<int>(height * outptr[5]);

      int w = xmax - xmin;
      int h = ymax - ymin;
      cv::Rect rec_clip =
          cv::Rect(xmin, ymin, w, h) & cv::Rect(0, 0, width, height);
      obj.rec = rec_clip;
      detect_result.push_back(obj);
    }
    outptr += 6;
  }
}

void Check::SecondStep()
{
  // Get Input Tensor
  std::unique_ptr<Tensor> input_tensor1(std::move(predictor_Second->GetInput(0)));
  int classify_w = 128;
  int classify_h = 128;
  input_tensor1->Resize({1, 3, classify_h, classify_w});
  auto *input_data = input_tensor1->mutable_data<float>();
  int detect_num = detect_result.size();
  std::vector<float> classify_mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> classify_scale = {1.f, 1.f, 1.f};

  json = "{ \"Num\":" + std::to_string(detect_num) + ",";
  json += "\"Data\": [";

  for (int i = 0; i < detect_num; ++i)
  {
    cv::Rect rec_clip = detect_result[i].rec;
    cv::Mat roi = crop_img(img, rec_clip, classify_w, classify_h);

    // Do PreProcess
    pre_process(roi, classify_w, classify_h, classify_mean, classify_scale, input_data, true);

    // Classification Model Run
    predictor_Second->Run();

    // Get Output Tensor
    std::unique_ptr<const Tensor> output_tensor1(std::move(predictor_Second->GetOutput(0)));
    auto *outptr = output_tensor1->data<float>();
    float prob = outptr[1];

    x = "\"x\":" + std::to_string(rec_clip.x) + ",";
    y = "\"y\":" + std::to_string(rec_clip.y) + ",";
    w = "\"width\":" + std::to_string(rec_clip.width) + ",";
    h = "\"height\":" + std::to_string(rec_clip.height) + ",";
    p = "\"prob\":" + std::to_string(prob);

    if (detect_num == 1 or i == detect_num - 1)
    {
      json += "{" + x + y + w + h + p + "}";
    }
    else
    {
      json += "{" + x + y + w + h + p + "},";
    }
  }

  json += "]}";
}

const char *Check::Sync(cv::Mat img_, int cols, int rows)
{
  img = img_;

  width = cols;
  height = rows;
  s_width = static_cast<int>(width * shrink);
  s_height = static_cast<int>(height * shrink);
  FirstStep();
  SecondStep();

  // int len = json.length();
  // data = (char *)malloc((len+1) * sizeof(char));
  // json.copy(data, len, 0);

  return json.data();
}

#endif //MASK_CHECK_H
