#include "opencl_engine.h"

template<>
cv::Mat sobel::OpenClEngine<3>::apply (const cv::Mat &img) const {
    const auto img_elements = img.rows * img.cols;
    const auto img_mem = sizeof(uchar) * img_elements;

    // Create buffers on the device
    cl::Buffer buffer_in(m_context, CL_MEM_READ_WRITE,img_mem);
    cl::Buffer buffer_cols(m_context, CL_MEM_READ_WRITE, sizeof(int));
    cl::Buffer buffer_out(m_context,CL_MEM_READ_WRITE,img_mem);
 
    // Create queue to which we will push commands for the device.
    cl::CommandQueue queue(m_context, m_device);

    constexpr uint8_t start_pixel = 1;

    std::vector<uchar> img_in(img_elements);
    for (int y = start_pixel ; y < img.rows - start_pixel - 1; y++) {
        for (int x = start_pixel; x < img.cols - start_pixel - 1; x++) {
            img_in[x*img.cols+y] = img.at<uchar>(cv::Point(x,y));
        }
    }
 
    // Write input image and cols to the device
    queue.enqueueWriteBuffer( buffer_in, CL_TRUE, 0, img_mem, &img_in[0]);
    queue.enqueueWriteBuffer( buffer_cols, CL_TRUE, 0, sizeof(int), &img.cols);
 
    // Send to 
    cl::Kernel kernel_sobel=cl::Kernel(m_program,"sobel_3");
    kernel_sobel.setArg(0,buffer_in);
    kernel_sobel.setArg(1,buffer_cols);
    kernel_sobel.setArg(2,buffer_out);
    queue.enqueueNDRangeKernel(kernel_sobel,cl::NullRange,cl::NDRange(img.cols, img.rows),cl::NullRange);
    queue.finish();

    std::vector<uchar> img_out(img_elements);

    // Read result from the output image
    queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, img_mem, &img_out[0]);
 
    cv::Mat result(img.rows, img.cols, img.type(), cv::Scalar{0});
    
    // Inner loop iterates over a row for better cache usage
    for (int y = start_pixel ; y < img.rows - start_pixel - 1; y++) {
        for (int x = start_pixel; x < img.cols - start_pixel - 1; x++) {
            result.at<uchar>(cv::Point(x,y)) = img_out[x * img.rows + y];
        }
    }

    return result;
}