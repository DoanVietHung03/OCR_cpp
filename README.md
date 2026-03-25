# 🚗 OCR_cpp - Real-Time License Plate Recognition System
OCR_cpp là một hệ thống nhận diện biển số xe (License Plate Recognition - LPR) thời gian thực được viết bằng C++. Dự án được thiết kế để xử lý song song nhiều luồng camera an ninh (CCTV), tích hợp các mô hình Deep Learning tiên tiến để phát hiện xe, theo dõi quỹ đạo, phát hiện biển số và đọc chữ/số (OCR) với độ chính xác cao.

Hệ thống tận dụng bộ tăng tốc phần cứng CUDA/cuDNN thông qua ONNX Runtime, cho phép chạy mượt mà và đạt mức FPS thời gian thực ngay cả trên các dòng GPU phổ thông (ví dụ: NVIDIA GeForce GTX 1650 Ti).

# ✨ Tính năng chính
⚡ Xử lý đa luồng (Multi-Camera Support): Xử lý đồng thời 2 luồng video CCTV một cách độc lập bằng std::async để tối ưu hóa tài nguyên CPU/GPU.

🏎️ Tích hợp Tracking thông minh: Ứng dụng thuật toán BYTETracker (kết hợp Kalman Filter) để theo dõi xe qua nhiều khung hình, giữ ID ổn định ngay cả khi xe bị khuất.

🧠 Cơ chế Cache OCR: Tự động lưu trữ (cache) kết quả đọc biển số tốt nhất của từng xe theo ID. Khi độ tự tin (confidence) đạt >80%, hệ thống tự động bỏ qua bước chạy mô hình OCR ở các frame tiếp theo để tăng tốc độ xử lý tổng thể.

🔍 Xử lý ảnh nâng cao: Tích hợp bộ lọc CLAHE để tự động cân bằng sáng, làm rõ nét chữ trong điều kiện cổng vào bị lóa sáng hoặc thiếu sáng.

📏 Hỗ trợ biển số đa định dạng: Tự động nhận diện và ghép chuỗi cho cả biển số dài (1 dòng) và biển số vuông (2 dòng) dựa trên tỷ lệ khung hình (aspect ratio) của biển.

# 🏗️ Kiến trúc Pipeline AI
Hệ thống sử dụng chuỗi mô hình (Pipeline) được tối ưu hóa dưới định dạng .onnx:

1. Vehicle Detection: YOLO11s (yolo11s.onnx) - Phát hiện ô tô, xe máy, xe buýt, xe tải.

2. Plate Detection: YOLOv9 (yolov9_detect_plate.onnx) - Xác định vị trí biển số trên vùng ảnh của xe.

3. OCR (Optical Character Recognition): PARSeq (parseq_2.onnx) - Đọc ký tự trên biển số.

# 💻 Yêu cầu hệ thống (Prerequisites)
## Để biên dịch và chạy dự án, hệ thống của bạn cần đáp ứng:

- Hệ điều hành: Windows 10/11.

- IDE: Visual Studio (khuyến nghị phiên bản 2019 hoặc 2022 để biên dịch CMake dễ dàng).

- Trình biên dịch: Hỗ trợ chuẩn C++17 trở lên.

- Phần cứng: GPU NVIDIA có hỗ trợ kiến trúc CUDA.

## Thư viện phụ thuộc:

- OpenCV: Đã được build kèm module CUDA.

- ONNX Runtime: Phiên bản hỗ trợ GPU (C++ API).

- Eigen 3: Phiên bản 3.4.0 (Dùng cho thư viện tính toán của BYTETracker).

# ⚙️ Hướng dẫn Cài đặt & Biên dịch (Build)
1. Chuẩn bị cấu trúc thư mục:

Đảm bảo bạn đã đặt các file mô hình .onnx vào thư mục weights/ và các video test vào thư mục CCTV/ trong gốc dự án.

2. Cấu hình đường dẫn thư viện:

Mở file CMakeLists.txt, bạn sẽ cần điều chỉnh lại các đường dẫn tuyệt đối sau đây cho khớp với máy tính thực thi (hiện tại đang được thiết lập sẵn cho môi trường phát triển gốc):

*** File CMake***
``set(OpenCV_DIR "D:/OpenCV_CUDA/build")
set(ONNXRUNTIME_DIR "D:/onnxruntime-gpu/onnxruntime-gpu")
target_include_directories(build_app PRIVATE "C:/Users/ACER/OneDrive/Desktop/OCR_cpp/eigen-3.4.0")``

3. Biên dịch với CMake (thông qua Visual Studio):

- Mở Visual Studio và chọn "Open a local folder" -> Trỏ đến thư mục chứa file CMakeLists.txt.

- Đợi CMake tự động generate cấu hình.

- Chọn target build_app.exe và nhấn Build (hoặc F5 để chạy thẳng).

*Lưu ý: Hệ thống CMake đã được cấu hình tự động sao chép thư mục weights, CCTV và các file DLL (kể cả dll của ONNX Runtime) vào thư mục build cuối cùng để bạn có thể chạy file .exe trực tiếp mà không bị lỗi thiếu thư viện.*

# 🚀 Cách thức hoạt động của Code
- Mã nguồn khởi tạo 2 instance của BYTETracker và 2 std::map đóng vai trò là Cache độc lập cho từng camera để tránh xung đột dữ liệu.

- Vòng lặp chính đọc frame, nếu là frame cần xử lý (nhảy cóc theo FRAME_SKIP_INTERVAL = 3 để tăng FPS), ứng dụng sẽ đẩy task vào std::async.

- Đầu ra được hiển thị trên một "LPR Security Dashboard" gộp cả 2 camera, đi kèm thông số FPS thực tế tại cổng. Nhấn ESC để tắt an toàn hệ thống và giải phóng bộ nhớ.
