#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <filesystem>
#include <iomanip>
#include <future>

#include "BYTETracker.h" 

namespace fs = std::filesystem;

// ================================
// CÁC HẰNG SỐ VÀ CẤU TRÚC DỮ LIỆU
// ================================
const std::string CHARSET = R"(0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)";
const std::map<int, std::string> target_vehicles = {
    {2, "car"}, {3, "motorcycle"}, {5, "bus"}, {7, "truck"}
};

// Cấu trúc lưu trữ kết quả OCR trong Cache
struct OCRResult {
    std::string text;
    float confidence;
    int update_count; // Số lần đã cập nhật để tránh update liên tục
};

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

// ====================================
// 1. CÁC HÀM TIỀN XỬ LÝ VÀ CHUẨN HÓA
// ====================================
std::string clean_plate_text(const std::string& input) {
    std::string output = "";
    for (char c : input) {
        if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) output += std::toupper(c);
    }
    return output;
}

std::string clean_bottom_line(const std::string& input) {
    std::string output = "";
    for (char c : input) { if (c >= '0' && c <= '9') output += c; }
    if (output.length() > 5) output = output.substr(0, 5);
    return output;
}

std::string clean_top_line(const std::string& input) {
    std::string output = clean_plate_text(input);
    if (output.length() > 5) output = output.substr(0, 5);
    return output;
}

cv::Mat preprocess_and_normalize_ocr(const cv::Mat& src, int target_w = 128, int target_h = 32) {
    if (src.empty()) return src;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
    cv::Mat rgb_img;
    cv::cvtColor(resized, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32FC3, 1.0 / 255.0);

    cv::Mat blob;
    cv::dnn::blobFromImage(rgb_img, blob, 1.0, cv::Size(), cv::Scalar(0, 0, 0), false, false);

    float mean_val = 0.5f, std_val = 0.5f;
    float* data = (float*)blob.data;
    for (int i = 0; i < blob.total(); ++i) data[i] = (data[i] - mean_val) / std_val;
    return blob;
}

// Tích hợp CLAHE giúp làm rõ nét chữ trong điều kiện lóa/tối ở cổng
cv::Mat enhance_plate_quality(const cv::Mat& src) {
    if (src.empty()) return src;
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    cv::Mat lab;
    cv::cvtColor(dst, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);

    cv::merge(lab_planes, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    return dst;
}

cv::Mat letterbox_yolo(const cv::Mat& source, int expected_width, int expected_height, float& ratio, int& pad_w, int& pad_h) {
    int h = source.rows, w = source.cols;
    ratio = std::min((float)expected_height / h, (float)expected_width / w);
    int new_unpad_w = int(std::round(w * ratio));
    int new_unpad_h = int(std::round(h * ratio));

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(new_unpad_w, new_unpad_h));

    pad_w = (expected_width - new_unpad_w) / 2;
    pad_h = (expected_height - new_unpad_h) / 2;

    cv::Mat padded = cv::Mat(expected_height, expected_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_unpad_w, new_unpad_h)));
    return padded;
}

// =============================
// 2. INFERENCE (YOLO & PARSEQ)
// =============================
std::vector<Detection> infer_yolo(Ort::Session& session, Ort::MemoryInfo& memory_info, const char** input_names, const char** output_names, const cv::Mat& img, float conf_thresh = 0.5f, const std::vector<int>& allowed_classes = {}) {
    auto input_info = session.GetInputTypeInfo(0);
    auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = tensor_info.GetShape();

    int input_h = (input_shape.size() >= 4 && input_shape[2] > 0) ? input_shape[2] : 640;
    int input_w = (input_shape.size() >= 4 && input_shape[3] > 0) ? input_shape[3] : 640;

    std::vector<int64_t> safe_input_shape = { 1, 3, input_h, input_w };
    float ratio; int pad_w, pad_h;
    cv::Mat padded_img = letterbox_yolo(img, input_w, input_h, ratio, pad_w, pad_h);

    cv::Mat blob;
    cv::dnn::blobFromImage(padded_img, blob, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), safe_input_shape.data(), safe_input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
    float* out_data = output_tensors[0].GetTensorMutableData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> class_ids;
    bool is_end2end_nms = (shape.size() == 3 && (shape[2] == 6 || shape[2] == 7) && shape[1] < 2000) || (shape.size() == 2 && (shape[1] == 6 || shape[1] == 7));

    if (is_end2end_nms) {
        int rows = (shape.size() == 3) ? shape[1] : shape[0];
        int dimensions = (shape.size() == 3) ? shape[2] : shape[1];
        int offset = (dimensions == 7) ? 1 : 0;
        for (int i = 0; i < rows; ++i) {
            float* row_ptr = out_data + i * dimensions + offset;
            float val1 = row_ptr[4], val2 = row_ptr[5], conf; int cls;
            if (std::abs(val1 - std::round(val1)) < 1e-5 && std::abs(val2 - std::round(val2)) >= 1e-5) { cls = (int)val1; conf = val2; }
            else if (std::abs(val1 - std::round(val1)) >= 1e-5 && std::abs(val2 - std::round(val2)) < 1e-5) { conf = val1; cls = (int)val2; }
            else { cls = (int)val1; conf = val2; }

            if (conf > conf_thresh) {
                if (allowed_classes.empty() || std::find(allowed_classes.begin(), allowed_classes.end(), cls) != allowed_classes.end()) {
                    boxes.push_back(cv::Rect(int((row_ptr[0] - pad_w) / ratio), int((row_ptr[1] - pad_h) / ratio), int((row_ptr[2] - row_ptr[0]) / ratio), int((row_ptr[3] - row_ptr[1]) / ratio)));
                    scores.push_back(conf); class_ids.push_back(cls);
                }
            }
        }
    }
    else {
        int dim1 = (shape.size() == 3) ? shape[1] : shape[0], dim2 = (shape.size() == 3) ? shape[2] : shape[1];
        cv::Mat output_mat(dim1, dim2, CV_32F, out_data);
        if (dim1 < dim2) output_mat = output_mat.t();

        for (int i = 0; i < output_mat.rows; ++i) {
            float* row_ptr = output_mat.ptr<float>(i);
            auto max_it = std::max_element(row_ptr + 4, row_ptr + output_mat.cols);
            if (*max_it > conf_thresh && (allowed_classes.empty() || std::find(allowed_classes.begin(), allowed_classes.end(), std::distance(row_ptr + 4, max_it)) != allowed_classes.end())) {
                boxes.push_back(cv::Rect(int(((row_ptr[0] - 0.5f * row_ptr[2]) - pad_w) / ratio), int(((row_ptr[1] - 0.5f * row_ptr[3]) - pad_h) / ratio), int(row_ptr[2] / ratio), int(row_ptr[3] / ratio)));
                scores.push_back(*max_it); class_ids.push_back(std::distance(row_ptr + 4, max_it));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, 0.4f, indices);
    std::vector<Detection> final_dets;
    for (int idx : indices) {
        cv::Rect safe_box = boxes[idx] & cv::Rect(0, 0, img.cols, img.rows);
        if (safe_box.area() > 0) final_dets.push_back({ safe_box, scores[idx], class_ids[idx] });
    }
    return final_dets;
}

// Hàm giải mã kết quả PARSEQ, tính confidence trung bình và loại bỏ ký tự không hợp lệ
std::string decode_parseq(const float* logits_data, int seq_len, int num_classes, float& out_confidence) {
    std::string result = "";
    float total_conf = 0.0f; int valid_chars = 0;
    for (int i = 0; i < seq_len; ++i) {
        float max_val = -1e9; int max_idx = -1;
        for (int j = 0; j < num_classes; ++j) {
            if (logits_data[i * num_classes + j] > max_val) { max_val = logits_data[i * num_classes + j]; max_idx = j; }
        }
        if (max_idx == 0) break;

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) sum_exp += std::exp(logits_data[i * num_classes + j] - max_val);
        float char_confidence = 1.0f / sum_exp;

        if (max_idx > 0 && max_idx <= CHARSET.length()) {
            result += CHARSET[max_idx - 1];
            total_conf += char_confidence; valid_chars++;
        }
    }
    out_confidence = (valid_chars > 0) ? (total_conf / valid_chars) : 0.0f;
    return result;
}

// =======================================================
// 3. HÀM XỬ LÝ 1 KHUNG HÌNH VỚI BYTETRACK & CACHE
// =======================================================
void process_single_frame(cv::Mat& img_bgr,
    Ort::Session& vehicle_session, Ort::Session& plate_session, Ort::Session& parseq_session,
    Ort::MemoryInfo& memory_info,
    const char** vehicle_in, const char** vehicle_out,
    const char** plate_in, const char** plate_out,
    const char** parseq_in, const char** parseq_out,
    const std::vector<int>& allowed_vehicle_ids,
    BYTETracker& tracker, std::map<int, OCRResult>& ocr_cache) {

    // 1. Nhận diện xe
    auto vehicle_dets = infer_yolo(vehicle_session, memory_info, vehicle_in, vehicle_out, img_bgr, 0.4f, allowed_vehicle_ids);

    // 2. Chuyển Detection sang định dạng Object của ByteTrack
    std::vector<Object> bt_objects;
    for (const auto& v : vehicle_dets) {
        Object obj;
        obj.rect = v.box;
        obj.prob = v.score;
        obj.label = v.class_id;
        bt_objects.push_back(obj);
    }

    // 3. Cập nhật Tracker
    std::vector<STrack> tracked_stracks = tracker.update(bt_objects);

    // 4. Xử lý OCR dựa trên Track ID
    for (const auto& track : tracked_stracks) {
        int track_id = track.track_id;
        std::vector<float> tlwh = track.tlwh; // Top, Left, Width, Height
        cv::Rect v_box(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);

        // Cắt cho an toàn trong viền ảnh
        v_box = v_box & cv::Rect(0, 0, img_bgr.cols, img_bgr.rows);
        if (v_box.width < 60 || v_box.height < 60) continue;

        std::string final_text = "";

        // KIỂM TRA CACHE: Nếu xe này đã đọc biển rõ (confidence > 80%), bỏ qua AI hoàn toàn!
        bool needs_ocr = true;
        if (ocr_cache.find(track_id) != ocr_cache.end()) {
            if (ocr_cache[track_id].confidence > 0.80f || ocr_cache[track_id].update_count >= 3) {
                needs_ocr = false;
                final_text = ocr_cache[track_id].text;
            }
        }

        if (needs_ocr) {
            // Nới rộng ROI xe 5%
            cv::Rect vehicle_roi(
                cv::Point(std::max(0, int(v_box.x - v_box.width * 0.05)), std::max(0, int(v_box.y - v_box.height * 0.05))),
                cv::Point(std::min(img_bgr.cols, int(v_box.x + v_box.width * 1.05)), std::min(img_bgr.rows, int(v_box.y + v_box.height * 1.05)))
            );

            auto plates_on_vehicle = infer_yolo(plate_session, memory_info, plate_in, plate_out, img_bgr(vehicle_roi), 0.5f);

            float best_plate_conf = 0.0f;

            for (const auto& p_det : plates_on_vehicle) {
                cv::Rect abs_plate_box(p_det.box.x + vehicle_roi.x, p_det.box.y + vehicle_roi.y, p_det.box.width, p_det.box.height);
                if (abs_plate_box.y > img_bgr.rows * 0.88 && abs_plate_box.x > img_bgr.cols * 0.60) continue;

                cv::Rect safe_plate_box = abs_plate_box & cv::Rect(0, 0, img_bgr.cols, img_bgr.rows);
                cv::Mat img_plate_raw = img_bgr(safe_plate_box);

                if (img_plate_raw.empty() || img_plate_raw.cols < 40 || img_plate_raw.rows < 20) continue;

                cv::Mat img_plate = enhance_plate_quality(img_plate_raw);
                if (img_plate.rows * 1.0 / img_plate.cols >= 1.5) cv::rotate(img_plate, img_plate, cv::ROTATE_90_CLOCKWISE);

                float ratio = img_plate.cols * 1.0 / img_plate.rows;
                float current_conf = 0.0f;
                std::string current_text = "";

                if (ratio > 2.2) {
                    cv::Mat blob_1line = preprocess_and_normalize_ocr(img_plate, 128, 32);
                    std::vector<int64_t> parseq_shape = { 1, 3, 32, 128 };
                    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob_1line.data, blob_1line.total(), parseq_shape.data(), parseq_shape.size());

                    auto outputs = parseq_session.Run(Ort::RunOptions{ nullptr }, parseq_in, &tensor, 1, parseq_out, 1);
                    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

                    current_text = clean_plate_text(decode_parseq(outputs[0].GetTensorMutableData<float>(), shape[1], shape[2], current_conf));
                }
                else {
                    cv::Mat img_top = img_plate(cv::Rect(0, 0, img_plate.cols, img_plate.rows * 0.60));
                    cv::Mat img_bottom = img_plate(cv::Rect(0, img_plate.rows * 0.40, img_plate.cols, img_plate.rows * 0.60));

                    cv::Mat blob_top = preprocess_and_normalize_ocr(img_top, 128, 32);
                    cv::Mat blob_bot = preprocess_and_normalize_ocr(img_bottom, 128, 32);
                    std::vector<int64_t> parseq_shape = { 1, 3, 32, 128 };

                    Ort::Value t_top = Ort::Value::CreateTensor<float>(memory_info, (float*)blob_top.data, blob_top.total(), parseq_shape.data(), parseq_shape.size());
                    Ort::Value t_bot = Ort::Value::CreateTensor<float>(memory_info, (float*)blob_bot.data, blob_bot.total(), parseq_shape.data(), parseq_shape.size());

                    auto out_top = parseq_session.Run(Ort::RunOptions{ nullptr }, parseq_in, &t_top, 1, parseq_out, 1);
                    auto out_bot = parseq_session.Run(Ort::RunOptions{ nullptr }, parseq_in, &t_bot, 1, parseq_out, 1);

                    auto shape_top = out_top[0].GetTensorTypeAndShapeInfo().GetShape();
                    auto shape_bot = out_bot[0].GetTensorTypeAndShapeInfo().GetShape();

                    float conf_top = 0.0f, conf_bot = 0.0f;
                    std::string text_top = clean_top_line(decode_parseq(out_top[0].GetTensorMutableData<float>(), shape_top[1], shape_top[2], conf_top));
                    std::string text_bot = clean_bottom_line(decode_parseq(out_bot[0].GetTensorMutableData<float>(), shape_bot[1], shape_bot[2], conf_bot));

                    std::string text_2_line = text_top + "-" + text_bot;
                    if (!text_2_line.empty() && text_2_line.front() == '-') text_2_line.erase(0, 1);
                    if (!text_2_line.empty() && text_2_line.back() == '-') text_2_line.pop_back();

                    current_text = text_2_line;
                    current_conf = (conf_top + conf_bot) / 2.0f;
                }

                // Lưu kết quả tốt nhất của xe này
                if (current_conf > best_plate_conf) {
                    best_plate_conf = current_conf;
                    final_text = current_text;
                }
            } 

            // Cập nhật Cache
            if (!final_text.empty() && final_text != "UNKNOWN") {
                if (ocr_cache.find(track_id) == ocr_cache.end()) {
                    ocr_cache[track_id] = { final_text, best_plate_conf, 1 };
                }
                else if (best_plate_conf > ocr_cache[track_id].confidence) {
                    ocr_cache[track_id].text = final_text;
                    ocr_cache[track_id].confidence = best_plate_conf;
                    ocr_cache[track_id].update_count++;
                }
            }
        } 

        // 5. Vẽ kết quả lên màn hình
        cv::rectangle(img_bgr, v_box, cv::Scalar(255, 0, 0), 2);

        // Vẽ Text ID và Biển số
        std::string display_text = "ID: " + std::to_string(track_id);
        if (!final_text.empty() && final_text != "UNKNOWN") {
            display_text += " | " + final_text;
        }

        cv::putText(img_bgr, display_text, cv::Point(v_box.x, std::max(0, v_box.y - 10)), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 4);
        cv::putText(img_bgr, display_text, cv::Point(v_box.x, std::max(0, v_box.y - 10)), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 4);
    }   
}

// =======================================================
// 4. MAIN 
// =======================================================
int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Pipeline_Inference");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(2); // Giới hạn thread để chạy mượt 2 luồng

        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        Ort::Session vehicle_session(env, L"weights/yolo11s.onnx", session_options);
        Ort::Session plate_session(env, L"weights/yolov9_detect_plate.onnx", session_options);
        Ort::Session parseq_session(env, L"weights/parseq_2.onnx", session_options);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;

        Ort::AllocatedStringPtr v_in_ptr = vehicle_session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr v_out_ptr = vehicle_session.GetOutputNameAllocated(0, allocator);
        const char* vehicle_in_names[] = { v_in_ptr.get() }; const char* vehicle_out_names[] = { v_out_ptr.get() };

        Ort::AllocatedStringPtr p_in_ptr = plate_session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr p_out_ptr = plate_session.GetOutputNameAllocated(0, allocator);
        const char* plate_in_names[] = { p_in_ptr.get() }; const char* plate_out_names[] = { p_out_ptr.get() };

        Ort::AllocatedStringPtr ocr_in_ptr = parseq_session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr ocr_out_ptr = parseq_session.GetOutputNameAllocated(0, allocator);
        const char* parseq_in_names[] = { ocr_in_ptr.get() }; const char* parseq_out_names[] = { ocr_out_ptr.get() };

        std::vector<int> allowed_vehicle_ids;
        for (const auto& pair : target_vehicles) allowed_vehicle_ids.push_back(pair.first);

        cv::VideoCapture cap1("CCTV/cong_K1.mp4");
        cv::VideoCapture cap2("CCTV/cong_K5.mp4");

        if (!cap1.isOpened() || !cap2.isOpened()) {
            std::cerr << "[LOI] Khong the doc video. Kiem tra lai duong dan file!\n";
            return -1;
        }

        cv::namedWindow("LPR Security Dashboard", cv::WINDOW_NORMAL);

        cv::Mat frame1, frame2;
        int frame_count = 0;
        const int FRAME_SKIP_INTERVAL = 3;
        cv::Mat last_display1, last_display2;

        // --- KHỞI TẠO BỘ TRACKER VÀ CACHE ĐỘC LẬP CHO 2 CAMERA ---
        int fps = 25; // Chỉnh theo FPS thực tế của camera
        int track_buffer = 30; // Giữ ID trong 30 frames nếu xe bị khuất
        BYTETracker tracker_cam1(fps, track_buffer);
        BYTETracker tracker_cam2(fps, track_buffer);

        std::map<int, OCRResult> cache_cam1;
        std::map<int, OCRResult> cache_cam2;
        // ---------------------------------------------------------

        std::cout << "[INFO] He thong LIVE VIEW dang hoat dong. Nhan 'ESC' de thoat...\n";

        while (true) {
            auto start_cycle = std::chrono::high_resolution_clock::now();

            bool ret1 = cap1.read(frame1);
            bool ret2 = cap2.read(frame2);

            if (!ret1 && !ret2) break;

            if (frame_count % FRAME_SKIP_INTERVAL == 0) {
                std::future<void> future1, future2;

                if (ret1) {
                    future1 = std::async(std::launch::async, [&]() {
                        process_single_frame(frame1, vehicle_session, plate_session, parseq_session, memory_info,
                            vehicle_in_names, vehicle_out_names, plate_in_names, plate_out_names,
                            parseq_in_names, parseq_out_names, allowed_vehicle_ids,
                            tracker_cam1, cache_cam1); // Truyền tracker và cache độc lập
                        });
                }
                if (ret2) {
                    future2 = std::async(std::launch::async, [&]() {
                        process_single_frame(frame2, vehicle_session, plate_session, parseq_session, memory_info,
                            vehicle_in_names, vehicle_out_names, plate_in_names, plate_out_names,
                            parseq_in_names, parseq_out_names, allowed_vehicle_ids,
                            tracker_cam2, cache_cam2); // Truyền tracker và cache độc lập
                        });
                }

                if (ret1 && future1.valid()) future1.wait();
                if (ret2 && future2.valid()) future2.wait();

                if (ret1) frame1.copyTo(last_display1);
                if (ret2) frame2.copyTo(last_display2);

                // Dọn dẹp cache định kỳ để tránh tràn RAM (xóa các xe đã đi qua quá lâu)
                if (frame_count % 300 == 0) {
                    cache_cam1.clear();
                    cache_cam2.clear();
                }
            }

            auto end_cycle = std::chrono::high_resolution_clock::now();
            double cycle_time_ms = std::chrono::duration<double, std::milli>(end_cycle - start_cycle).count();
            double display_fps = 1000.0 / cycle_time_ms;
            std::stringstream fps_str; 
            fps_str << "FPS: " << std::fixed << std::setprecision(1) << display_fps;

            cv::Mat dash1, dash2;

            if (ret1) {
                cv::Mat disp1 = last_display1.empty() ? frame1.clone() : last_display1.clone();
                cv::putText(disp1, "CAM 1 | " + fps_str.str(), cv::Point(20, disp1.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
                cv::putText(disp1, "CAM 1 | " + fps_str.str(), cv::Point(20, disp1.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
                cv::resize(disp1, dash1, cv::Size(960, 540));
            }
            else { dash1 = cv::Mat::zeros(cv::Size(960, 540), CV_8UC3); }

            if (ret2) {
                cv::Mat disp2 = last_display2.empty() ? frame2.clone() : last_display2.clone();
                cv::putText(disp2, "CAM 2 | " + fps_str.str(), cv::Point(20, disp2.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
                cv::putText(disp2, "CAM 2 | " + fps_str.str(), cv::Point(20, disp2.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
                cv::resize(disp2, dash2, cv::Size(960, 540));
            }
            else { dash2 = cv::Mat::zeros(cv::Size(960, 540), CV_8UC3); }

            cv::Mat dashboard;
            cv::hconcat(dash1, dash2, dashboard);
            cv::imshow("LPR Security Dashboard", dashboard);

            frame_count++;
            if (cv::waitKey(1) == 27) {
                std::cout << "[INFO] Nguoi dung an ESC. Tat he thong...\n";
                break;
            }
        }

        cap1.release();
        cap2.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "\n[LOI NGHIEM TRONG]: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}