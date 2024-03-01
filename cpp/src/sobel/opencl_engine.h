#include <iostream>
#include <optional>

#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION  120

#include <CL/opencl.hpp>
#include <opencv2/core.hpp>

namespace sobel {
    template<const uint8_t K>
    class OpenClEngine {
        public:

        /// @brief Checks whether this engine is available 
        /// @return nullopt if everything is alright, otherwise returns an string of what happened
        static std::optional<OpenClEngine> try_create() {
            // Obtain plaftorms
            std::vector<cl::Platform> all_platforms;
            cl::Platform::get(&all_platforms);
            if(all_platforms.size()==0){
                std::cout<<" No platforms found. Check OpenCL installation!\n";
                return std::nullopt;
            }

            // We'll just use the first available platform
            cl::Platform default_platform=all_platforms[0];
            std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

            //get default device of the default platform
            std::vector<cl::Device> all_devices;
            default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            if (all_devices.size()==0) {
                std::cout<<" No devices found. Check OpenCL installation!\n";
                return std::nullopt;
            }

            cl::Device default_device=all_devices[0];
            std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
 
            cl::Context context({default_device});

            cl::Program::Sources sources;
 
            // kernel calculates for each element C=A+B
            std::string kernel_code=
                    "   void kernel sobel_3(global const uchar* img_in, const int cols, global uchar* img_out){       "
                    "       const int x = get_global_id(0);"
                    "       const int y = get_global_id(1);"
                    "       const short mag_x = "
                    "           (-1 * img_in[(x-1)*cols + y-1]) + img_in[(x+1)*cols + y-1] +"
                    "           (-2 * img_in[(x-1)*cols + y]) + (2 * img_in[(x+1)*cols + y]) +"
                    "           (-1 * img_in[(x-1)*cols + y+1]) + img_in[(x+1)*cols + y+1];"

                    "       const short mag_y = "
                    "           (-1 * img_in[(x-1)*cols + y-1]) + (-2 * img_in[x * cols + y-1]) + (-1 * img_in[(x+1)*cols + y-1]) +"
                    "           img_in[(x-1)*cols + y+1] + (2 * img_in[x*cols + y+1]) + img_in[(x+1)*cols+y+1];"
                    
                    "       img_out[x*cols + y] = (uchar) sqrt( (float)((mag_x*mag_x) + (mag_y*mag_y)));"
                    "   }";
            sources.push_back({kernel_code.c_str(),kernel_code.length()});
        
            cl::Program program(context,sources);
            if(program.build({default_device})!=CL_SUCCESS){
                std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
                exit(1);
            }

            return OpenClEngine(default_device,context, program);
        }

        /// @brief Applies the filter to an incoming grayscale image
        /// @param img The original image
        /// @return The image with the filter applied
        cv::Mat apply(const cv::Mat &img) const;

        private:
            OpenClEngine(cl::Device device, cl::Context context, cl::Program program): m_device(device), m_context(context), m_program(program) {
            }

            cl::Device m_device;
            cl::Context m_context;
            cl::Program m_program;
    };
}