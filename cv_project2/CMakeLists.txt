cmake_minimum_required(VERSION 2.4)
PROJECT(cv_project2)

# paths 
INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(/usr/local/include)
LINK_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(/usr/lib)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})


#ADD_EXECUTABLE(homo src/project2_homo.cpp)
#TARGET_LINK_LIBRARIES(homo ${OpenCV_LIBS})
ADD_EXECUTABLE(guass src/project2_guass.cpp)
TARGET_LINK_LIBRARIES(guass ${OpenCV_LIBS})
SET(CMAKE_BUILD_TYPE Release)
