#include "software_engine.h"

// Note: Each variation of the sobel filter is independently implemented, since each version
// differs greatly on how they can be optimized.
template<> uint8_t sobel::SoftwareEngine<3>::calc_for_pixel(const cv::Mat &org_pic, int x, int y) {
    assert(x > 0 && y > 0 && x < (org_pic.cols - 2) && y < (org_pic.rows - 2));

    constexpr int16_t kernelx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    constexpr int16_t kernely[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    const auto img_gray_y_before = org_pic.ptr<uchar>(y-1);
    const auto img_gray_y = org_pic.ptr<uchar>(y);
    const auto img_gray_y_next = org_pic.ptr<uchar>(y+1);

    const int16_t mag_x = (kernelx[0][0] * img_gray_y_before[x-1]) + (kernelx[0][1] * img_gray_y_before[x]) + (kernelx[0][2] * img_gray_y_before[x+1]) +
            (kernelx[1][0] * img_gray_y[x-1])   + (kernelx[1][1] * img_gray_y[x])   + (kernelx[1][2] * img_gray_y[x+1]) +
            (kernelx[2][0] * img_gray_y_next[x-1]) + (kernelx[2][1] * img_gray_y_next[x]) + (kernelx[2][2] * img_gray_y_next[x+1]);

    const int16_t mag_y = (kernely[0][0] * img_gray_y_before[x-1]) + (kernely[0][1] * img_gray_y_before[x]) + (kernely[0][2] * img_gray_y_before[x+1]) +
            (kernely[2][0] * img_gray_y_next[x-1]) + (kernely[2][1] * img_gray_y_next[x]) + (kernely[2][2] * img_gray_y_next[x+1]);

    return sqrt((mag_x*mag_x) + (mag_y*mag_y));
}



