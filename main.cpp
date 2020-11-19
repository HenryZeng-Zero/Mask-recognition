//
// Created by HenryZeng on 2020/10/30.
//

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "Check2.h"

Check Video;

// extern "C"
// {
//     void LoadModel(char *detect_model_file, char *classify_model_file)
//     {
//         std::cout<<"detect_model:"<<detect_model_file<<std::endl<<"classify_model:"<<classify_model_file<<std::endl;
//         Video.LoadModel(std::string(detect_model_file), std::string(classify_model_file));
//     }

//     char* VC(uchar *img_data, int rows, int cols)
//     {
//         cv::Mat src = cv::Mat(cv::Size(cols, rows), CV_8UC3, cv::Scalar(255, 255, 255));
//         src.data = img_data;

//         cv::imshow("Show",src);
//         cv::waitKey(0);

//         return Video.Sync(src);
//     }
// }

int main()
{
    std::string a = "/home/pi/Mask/Data/mask_detector.nb";
    std::string b = "/home/pi/Mask/Data/pyramidbox_lite.nb";
    Video.LoadModel(b,a);
    cv::Mat M = cv::imread("/home/pi/Mask/Data/test_img.jpg");
    Video.Sync(&M);

}