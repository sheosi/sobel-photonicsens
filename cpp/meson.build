project('sobel-photonicsens', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3',
                     'cpp_std=c++17'])

opencv = dependency('opencv4')
opencl = dependency('OpenCL')
exe = executable('sobel_photonicsens', 
  ['src/sobel_photonicsens.cpp', 'src/sobel/software_engine.cpp', 'src/sobel/opencl_engine.cpp'],
  install : true, dependencies: [opencv, opencl],
  cpp_args : '-std=c++17'
)

test('basic', exe)
