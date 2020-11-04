//
// Created by HenryZeng on 2020/10/30.
//

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "Check.h"

Check Video;

void LoadModel(std::string detect_model_file,std::string classify_model_file){
    Video.LoadModel(detect_model_file, classify_model_file);
}

std::string Input(cv::Mat img){
    return Video.Sync(img);
}