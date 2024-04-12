#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ObjectDetector.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_cellgraphics_onnxobjectdetection_MainActivity_initDetector(JNIEnv* env, jobject, jobject assetManager) {
    char* buffer = nullptr;
    long size = 0;
    if (!(env->IsSameObject(assetManager, nullptr))) {
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
        AAsset* asset = AAssetManager_open(mgr, "ssd_mobilenet_v1_12.onnx", AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);
        size = AAsset_getLength(asset);
        buffer = (char*)malloc(size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }
    std::unique_ptr<Ort::Env> environment(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "ortenv"));
    auto res = (jlong)new ObjectDetector(environment, buffer, size);
    free(buffer);
    return res;
}

void rotateMat(cv::Mat &matImage, int rotation) {
    if (rotation == 90) {
        cv::transpose(matImage, matImage);
        cv::flip(matImage, matImage, 1); //transpose+flip(1)=CW
    } else if (rotation == 270) {
        cv::transpose(matImage, matImage);
        cv::flip(matImage, matImage, 0); //transpose+flip(0)=CCW
    } else if (rotation == 180) {
        cv::flip(matImage, matImage, -1); //flip(-1)=180
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_cellgraphics_onnxobjectdetection_MainActivity_detect(JNIEnv *env, jobject, jlong detectorAddr, jbyteArray src, int width, int height, int rotation) {
    // Frame bytes to Mat
    jbyte* yuv = env->GetByteArrayElements(src, nullptr);
    cv::Mat yuvFrame(height + height / 2, width, CV_8UC1, yuv);
    cv::Mat frame(height, width, CV_8UC4);
    cv::cvtColor(yuvFrame, frame, cv::COLOR_YUV2BGRA_NV21);
    rotateMat(frame, rotation);
    env->ReleaseByteArrayElements(src, yuv, 0);

    // Detect
    auto detector = (ObjectDetector*)detectorAddr;
    DetectResult* res = detector->detect(frame);

    // Encode each detection as 6 numbers (label, score, xmin, xmax, ymin, ymax)
    int resArrLen = detector->DETECT_NUM * 6;
    jfloat jres[resArrLen];
    for (int i = 0; i < detector->DETECT_NUM; ++i) {
        jres[i * 6] = (jfloat)res[i].label;
        jres[i * 6 + 1] = res[i].score;
        jres[i * 6 + 2] = res[i].xmin;
        jres[i * 6 + 3] = res[i].xmax;
        jres[i * 6 + 4] = res[i].ymin;
        jres[i * 6 + 5] = res[i].ymax;
    }
    jfloatArray detections = env->NewFloatArray(resArrLen);
    env->SetFloatArrayRegion(detections, 0, resArrLen, jres);
    return detections;
}
