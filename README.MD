# Filtro Sobel

*cpp*: Implementación en c++ con opencl

*rs*: Implementación en Rust básica


Puntos a considerar:

La versión de C++ tiene tanto versión en Software cómo en OpenCL, sin embargo, la versión de OpenCl sufre un crash durante su ejecución (se puede hacer la prueba seteando force_cpu a false).

La versión de Rust solo emite imágenes en negro, pero parece que sí está haciendo todo el recorrido de la imágen en una fracción del tiempo (135ms en Rust vs 400ms en C++).