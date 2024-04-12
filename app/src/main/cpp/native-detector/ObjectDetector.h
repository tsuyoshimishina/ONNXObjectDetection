//
// Created by tmish on 2024/04/11.
//

#ifndef ONNX_OBJECT_DETECTION_OBJECTDETECTOR_H
#define ONNX_OBJECT_DETECTION_OBJECTDETECTOR_H

#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <cinttypes>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "nnapi_provider_factory.h"

#include <android/log.h>
#define LOG_TAG "onnx-object-detection"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct DetectResult {
    int label = -1;
    float score = 0.0f;
    float ymin = 0.0f;
    float xmin = 0.0f;
    float ymax = 0.0f;
    float xmax = 0.0f;
};

class ObjectDetector {
public:
    ObjectDetector(std::unique_ptr<Ort::Env>& env, const char *model, long modelSize);
    ~ObjectDetector() { }
    DetectResult* detect(const cv::Mat& src);
    const int DETECT_NUM = 3;

private:
    const int DETECTION_MODEL_SIZE = 300;
    const int DETECTION_MODEL_CNLS = 3;
    const float IMAGE_MEAN = 128.0f;
    const float IMAGE_STD = 128.0f;

    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::Session> m_ortSession;
    Ort::Value m_ortInputTensor{nullptr};
    std::string m_inputNodeName;
    std::unique_ptr<uint8_t[]> m_inputData;
    std::vector<std::string> m_outputNodeNames;

    void initDetectionModel(const char* model, long modelSize);
};

#endif //ONNX_OBJECT_DETECTION_OBJECTDETECTOR_H
