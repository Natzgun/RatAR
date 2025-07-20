cmake_minimum_required(VERSION 3.10)
project(OpenGL-project)

set(CMAKE_CXX_STANDARD 17)

# Agrega tu directorio de headers (glad, etc.)
include_directories(include)

# GLAD como biblioteca
add_library(glad src/glad.c)
target_include_directories(glad PUBLIC include)

# GLFW
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})

# ASSIMP
find_package(assimp REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#glew
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})



# Ejecutable
add_executable(OpenGL-project src/main.cc)

# Vincula todas las bibliotecas necesarias
target_link_libraries(OpenGL-project 
    glad 
    ${GLFW_LIBRARIES} 
    ${ASSIMP_LIBRARIES}
    ${OpenCV_LIBS}
    dl 
    GL
)
