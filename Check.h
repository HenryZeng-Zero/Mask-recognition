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

using namespace paddle::lite_api;  // NOLINT

struct Object {
    int batch_id;
    cv::Rect rec;
    int class_id;
    float prob;
};

class Check {
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

    std::string mode = "show";
public:
    void LoadModel(const std::string &det_model_file, const std::string &class_model_file);

    void FirstStep();

    void SecondStep();

    void setMode(const std::string &mode_);

    cv::Mat Sync(cv::Mat *img_);
};

void Check::LoadModel(const std::string &det_model_file, const std::string &class_model_file) {
    MobileConfig config;
    config.set_model_from_file(det_model_file);

    // Create Predictor For Detction Model
    predictor_First = CreatePaddlePredictor<MobileConfig>(config);
    std::cout << "Load detecion model succeed." << std::endl;

    config.set_model_from_file(class_model_file);

    // Create Predictor For Classification Model
    predictor_Second = CreatePaddlePredictor<MobileConfig>(config);
    std::cout << "Load classification model succeed." << std::endl;
}

void neon_mean_scale(const float *din,
                     float *dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
    if (mean.size() != 3 || scale.size() != 3) {
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
    for (; i < size - 3; i += 4) {
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
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
    }
}

void pre_process(const cv::Mat &img,
                 int width,
                 int height,
                 const std::vector<float> &mean,
                 const std::vector<float> &scale,
                 float *data,
                 bool is_scale = false) {
    cv::Mat resized_img;
    if (img.cols != width || img.rows != height) {
        cv::resize(
                img, resized_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
    } else {
        resized_img = img;
    }
    cv::Mat imgf;
    float scale_factor = is_scale ? 1.f / 256 : 1.f;
    resized_img.convertTo(imgf, CV_32FC3, scale_factor);
    const float *dimg = reinterpret_cast<const float *>(imgf.data);
    neon_mean_scale(dimg, data, width * height, mean, scale);
}

int64_t ShapeProduction(const shape_t &shape) {
    int64_t res = 1;
    for (auto i : shape) res *= i;
    return res;
}

void Check::FirstStep() {
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
    for (int i = 0; i < out_len / 6; ++i) {
        if (outptr[1] >= detect_threshold) {
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

cv::Mat crop_img(const cv::Mat &img,
                 cv::Rect rec,
                 int res_width,
                 int res_height) {
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

void Check::SecondStep() {
    // Get Input Tensor
    std::unique_ptr<Tensor> input_tensor1(std::move(predictor_Second->GetInput(0)));
    int classify_w = 128;
    int classify_h = 128;
    input_tensor1->Resize({1, 3, classify_h, classify_w});
    auto *input_data = input_tensor1->mutable_data<float>();
    int detect_num = detect_result.size();
    std::vector<float> classify_mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> classify_scale = {1.f, 1.f, 1.f};

    for (int i = 0; i < detect_num; ++i) {
        cv::Rect rec_clip = detect_result[i].rec;
        cv::Mat roi = crop_img(img, rec_clip, classify_w, classify_h);

        // uncomment two lines below, save roi img to disk
        // std::string roi_name = "roi_" + paddle::lite::to_string(i)
        // + ".jpg";
        // imwrite(roi_name, roi);

        // Do PreProcess
        pre_process(roi, classify_w, classify_h, classify_mean, classify_scale, input_data, true);

        // Classification Model Run
        predictor_Second->Run();

        // Get Output Tensor
        std::unique_ptr<const Tensor> output_tensor1(std::move(predictor_Second->GetOutput(0)));
        auto *outptr = output_tensor1->data<float>();
        float prob = outptr[1];
        if(mode == "show"){
            // Draw Detection and Classification Results
            bool flag_mask = prob > 0.8f;
            cv::Scalar roi_color;
            std::string text;
            if (flag_mask) {
                text = "MASK:  ";
                roi_color = cv::Scalar(0, 255, 0);
            } else {
                text = "NO MASK:  ";
                roi_color = cv::Scalar(0, 0, 255);
                prob = 1 - prob;
            }
            std::string prob_str = std::to_string(prob * 100);
            int point_idx = prob_str.find_last_of(".");

            text += prob_str.substr(0, point_idx + 3) + "%";
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.38;
            float thickness = 1;
            cv::Size text_size =
                    cv::getTextSize(text, font_face, font_scale, thickness, nullptr);

            int top_space = std::max(0.35 * text_size.height, 2.0);
            int bottom_space = top_space + 2;
            int right_space = 0.05 * text_size.width;
            int back_width = text_size.width + right_space;
            int back_height = text_size.height + top_space + bottom_space;

            // Configure text background
            cv::Rect text_back =
                    cv::Rect(rec_clip.x, rec_clip.y - back_height, back_width, back_height);

            // Draw roi object, text, and background
            cv::rectangle(img, rec_clip, roi_color, 1);
            cv::rectangle(img, text_back, cv::Scalar(225, 225, 225), -1);
            cv::Point origin;
            origin.x = rec_clip.x;
            origin.y = rec_clip.y - bottom_space;

            cv::putText(img,
                        text,
                        origin,
                        font_face,
                        font_scale,
                        cv::Scalar(0, 0, 0),
                        thickness);

            std::cout << "detect face, location: x=" << rec_clip.x
                      << ", y=" << rec_clip.y << ", width=" << rec_clip.width
                      << ", height=" << rec_clip.height << ", wear mask: " << flag_mask
                      << ", prob: " << prob << std::endl;
        } else if (mode == "api"){
            bool flag_mask = prob > 0.8f;

            std::cout << "detect face, location: x=" << rec_clip.x
                      << ", y=" << rec_clip.y << ", width=" << rec_clip.width
                      << ", height=" << rec_clip.height << ", wear_mask: " << bool(flag_mask)
                      << ", prob: " << prob << std::endl;
        }

    }

}

cv::Mat Check::Sync(cv::Mat *img_) {
    img = std::move(*img_);

    width = img.cols;
    height = img.rows;
    s_width = static_cast<int>(width * shrink);
    s_height = static_cast<int>(height * shrink);
    FirstStep();
    SecondStep();

    return img;
}

void Check::setMode(const std::string &mode_) {
    // show : show opencv test window
    // api : back the json data
    mode = mode_;
}

#endif //MASK_CHECK_H
