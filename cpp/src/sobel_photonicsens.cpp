#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "sobel/opencl_engine.h"
#include "sobel/software_engine.h"

// Configuration
constexpr bool show_window = false;
constexpr bool force_cpu = true; // OpenCL implementation is not working properly
constexpr uint8_t kernel_size = 3; // Note: in this version only a kernel of size 3 is implemented

// Code
int main(int argc, char **argv) {
    if(argc != 3) {
        std::cout << argv[0] <<  "Usage: sobel-photonicsens input output" << std::endl;
        return 1;
    }
    
    std::string in_image_path = argv[1];
    cv::Mat img = cv::imread(in_image_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    // Just give the pixels some initial value, is as fast as accessing it later
    cv::Mat sobel;
    
    if (force_cpu) {
        const auto soft_engine = sobel::SoftwareEngine<kernel_size>::try_create();
        sobel = soft_engine->apply(img);
    }
    else if (const auto ocl_engine = sobel::OpenClEngine<kernel_size>::try_create()){
        sobel = ocl_engine->apply(img);
    } 
    else {
        const auto soft_engine = sobel::SoftwareEngine<kernel_size>::try_create();
        sobel = soft_engine->apply(img);
    }

    if (show_window) {
        cv::imshow("Display window", sobel);
    }

    std::string out_image_path = argv[2];
    cv::imwrite(out_image_path, sobel);

    return 0;
}
