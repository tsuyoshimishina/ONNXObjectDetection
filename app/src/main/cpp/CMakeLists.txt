# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.19)

# Declares and names the project.

project("onnxobjectdetection")

# OpenCV
set(OpenCV_STATIC ON)
set(OpenCV_DIR $ENV{OPENCV_ANDROID}/sdk/native/jni)
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(Distribution_DIR ${CMAKE_SOURCE_DIR}/../../../../distribution)
add_library(libonnxruntime SHARED IMPORTED)
set_target_properties(libonnxruntime PROPERTIES IMPORTED_LOCATION ${Distribution_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        onnxobjectdetection

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp
        ./native-detector/ObjectDetector.cpp
)

target_include_directories(
        onnxobjectdetection PRIVATE
        ${Distribution_DIR}/include
        ./native-detector
)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# android for AssetManager functionality
find_library(android-lib android)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        onnxobjectdetection

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        ${android-lib}

        libonnxruntime
        ${OpenCV_LIBS}
)
