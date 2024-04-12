//
// Created by tmish on 2024/04/11.
//

#include "ObjectDetector.h"

ObjectDetector::ObjectDetector(std::unique_ptr<Ort::Env>& env, const char *model, long modelSize) : m_ortEnv(std::move(env))
{
    initDetectionModel(model, modelSize);
}

void ObjectDetector::initDetectionModel(const char *model, long modelSize)
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if 0
    uint32_t flags = NNAPI_FLAG_CPU_DISABLED;
#else
    uint32_t flags = 0;
#endif
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, flags));
    m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, model, modelSize, sessionOptions);
    assert(m_ortSession->GetInputCount() == 1);
    Ort::AllocatorWithDefaultOptions allocator;
    m_inputNodeName = m_ortSession->GetInputNameAllocated(0, allocator).get();
    auto inputTypeInfo = m_ortSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputDataType = inputTensorInfo.GetElementType();
    assert(inputDataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    std::vector<int64_t> inputNodeShape { 1, DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE, DETECTION_MODEL_CNLS };
    int64_t inputTensorSize = DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    m_inputData = std::make_unique<uint8_t[]>(inputTensorSize);
    m_ortInputTensor = Ort::Value::CreateTensor<uint8_t>(memoryInfo, m_inputData.get(), inputTensorSize, inputNodeShape.data(), inputNodeShape.size());
    assert(m_ortInputTensor.IsTensor());

    assert(m_ortSession->GetOutputCount() == 4);
    for (int i = 0; i < 4; ++i)
        m_outputNodeNames.push_back(m_ortSession->GetOutputNameAllocated(i, allocator).get());
}

DetectResult* ObjectDetector::detect(const cv::Mat& src)
{
    DetectResult res[DETECT_NUM];

    cv::Mat image;
    cv::resize(src, image, cv::Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, cv::INTER_AREA);
    int cnls = image.type();
    if (cnls == CV_8UC1)
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    else if (cnls == CV_8UC4)
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);

    // Copy image into input tensor
    memcpy(m_inputData.get(), image.data, DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);

    std::vector<const char*> inames { m_inputNodeName.c_str() }, onames;
    for (int i = 0; i < m_outputNodeNames.size(); ++i)
        onames.push_back(m_outputNodeNames[i].c_str());

    auto start = std::chrono::high_resolution_clock::now();
    auto outputTensors = m_ortSession->Run(Ort::RunOptions{nullptr}, inames.data(), &m_ortInputTensor, 1, onames.data(), onames.size());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    LOGI("Ort::Session::Run() takes %" PRId64 " milli seconds \n", duration.count());
    assert(outputTensors.size() == 4);

    const float* detectionBoxes = outputTensors[0].GetTensorMutableData<float>();
    const float* detectionClasses = outputTensors[1].GetTensorMutableData<float>();
    const float* detectionScores = outputTensors[2].GetTensorMutableData<float>();
    const int numDetections = (int)outputTensors[3].GetTensorMutableData<float>()[0];

    for (int i = 0; i < numDetections && i < DETECT_NUM; ++i) {
        res[i].score = detectionScores[i];
        res[i].label = (int)detectionClasses[i];

        // Get the bbox, make sure its not out of the image bounds, and scale up to src image size
        res[i].ymin = std::fmax(0.0f, detectionBoxes[4 * i] * src.rows);
        res[i].xmin = std::fmax(0.0f, detectionBoxes[4 * i + 1] * src.cols);
        res[i].ymax = std::fmin(float(src.rows - 1), detectionBoxes[4 * i + 2] * src.rows);
        res[i].xmax = std::fmin(float(src.cols - 1), detectionBoxes[4 * i + 3] * src.cols);
    }
    return res;
}
