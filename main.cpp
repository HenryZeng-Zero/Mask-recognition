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

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "[ERROR] usage: " << argv[0]
                  << " detction_model_file classification_model_file image_path\n";
        exit(1);
    }
     = argv[1];
     = argv[2];

    // std::string detect_model_file = "/home/pi/Mask/Data/pyramidbox_lite.nb";
    // std::string classify_model_file = "/home/pi/Mask/Data/mask_detector.nb";

    

    return 0;
}
