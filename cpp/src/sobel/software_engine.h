#include <optional>

#include <opencv2/core.hpp>

namespace sobel {
    template<const uint8_t K>
    class SoftwareEngine {
        public:

        /// @brief Creates one instance of this engine if possible, this is here just to keep consistency. 
        /// @return Always returns a SoftwareEngine.
        static std::optional<SoftwareEngine> try_create() {
            return SoftwareEngine();
        }

        /// @brief Applies the filter to an incoming grayscale image
        /// @param img The original image
        /// @return The image with the filter applied
        cv::Mat apply(const cv::Mat &img) const {
            cv::Mat result(img.rows,img.cols, img.type(), cv::Scalar{0});
            constexpr uint8_t start_pixel = (K + 1)/2 - 1;

            // Inner loop iterates over a row for better cache usage
            for (int y = start_pixel ; y < img.rows - start_pixel - 1; y++) {
                for (int x = start_pixel; x < img.cols - start_pixel - 1; x++) {
                    result.at<uchar>(cv::Point(x,y)) = calc_for_pixel(img, x,y);
                }
            }

            return result;
        }

        private:
            static uint8_t calc_for_pixel(const cv::Mat &org_pic, int x, int y);
    };
}