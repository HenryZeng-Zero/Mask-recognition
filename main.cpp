//
// Created by HenryZeng on 2020/10/30.
//

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "Check.h"

Check Video;

extern "C"
{
    void LoadModel(char *detect_model_file, char *classify_model_file)
    {
        std::cout << "detect_model:" << detect_model_file << std::endl
                  << "classify_model:" << classify_model_file << std::endl;
        Video.LoadModel(std::string(detect_model_file), std::string(classify_model_file));
    }

    const char *VC(uchar *img_data, int rows, int cols)
    {
        cv::Mat src = cv::Mat(cv::Size(cols, rows), CV_8UC3, cv::Scalar(255, 255, 255));
        src.data = img_data;
        return Video.Sync(src,cols,rows);
    }
}