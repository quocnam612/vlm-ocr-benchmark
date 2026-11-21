# VLM / OCR Benchmark trên OPRB

Đánh giá năm mô hình huấn luyện sẵn trên 9.000 ảnh test của
[OPRB](https://huggingface.co/datasets/kpurkayastha/OPRB), theo [LabOCR.pdf](LabOCR.pdf).
GT chỉ dùng để chấm điểm; không huấn luyện hay tinh chỉnh trên test.

[Báo cáo PDF tiếng Việt — 3 trang A4](latex_report/main.pdf) ·
[Mã LaTeX và cách tái tạo](latex_report/README.md).

## Kết quả chạy trên Google Drive

- [docTR](https://drive.google.com/drive/folders/1vosBxRMPkBXYH8vZbonjkj5QCFsF_Zy8?usp=sharing)
- [EasyOCR](https://drive.google.com/drive/folders/1FuShUqY_s74tVkic_I_YohYlBUN1PYTE?usp=sharing)
- [Florence](https://drive.google.com/drive/folders/1wB9eSuFXprUlT02RjaxQ2TkC1kDmldYf?usp=sharing)
- [PaddleOCR](https://drive.google.com/drive/folders/1YRyvsGkRrzAg2VqZ2uAJkeus6ThK4WzA?usp=sharing)
- [Tesseract](https://drive.google.com/drive/folders/1ey57vggKSCMPVbHZfNsQTWAxhwMR9bcX?usp=sharing)

## Mô hình

| Mô hình | Notebook |
| --- | --- |
| Florence-2-base | [Florence](OPRB_Florence_2_Base_Standard_Decoding.ipynb) |
| CRAFT + english_g2 | [EasyOCR](OPRB_EasyOCR_English_Test_Evaluation.ipynb) |
| DB / CRNN MobileNetV3 | [docTR](OPRB_docTR_MobileNet_Test_Evaluation.ipynb) |
| PP-OCRv5 mobile, nhận dạng tiếng Anh | [PaddleOCR](OPRB_PaddleOCR_PP_OCRv5_Mobile_English_Test_Evaluation.ipynb) |
| English tessdata_fast | [Tesseract](OPRB_Tesseract_English_Fast_Test_Evaluation.ipynb) |

Năm mô hình đã duyệt đủ 9.000 ảnh. Loại chung một GT chỉ chứa nhãn cấu trúc,
các bảng báo cáo dùng **cùng 8.999 ảnh** (6.000 text, 2.999 nontext).
Florence hiện đã hoàn tất, không còn là kết quả một phần. Qwen không nằm trong
benchmark này.

## Kết quả trên 8.999 ảnh chung

Trung bình đều theo ảnh; các tỷ lệ dùng %, thời gian dùng giây/ảnh.
Florence/EasyOCR/docTR/PaddleOCR dùng một Tesla T4; Tesseract dùng CPU Xeon.

| Mô hình | CER ↓ | WER ↓ | P ↑ | R ↑ | F1 ↑ | Giây/ảnh ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence | 61,27 | 79,33 | 53,41 | 31,69 | 38,55 | 5,597 |
| EasyOCR | 57,60 | 82,42 | 43,86 | 31,96 | 36,51 | 1,363 |
| docTR | 47,86 | 62,89 | 78,79 | 68,86 | 72,76 | 0,601 |
| PaddleOCR | 33,97 | 43,25 | 85,17 | 72,97 | 77,75 | 0,403 |
| Tesseract | 47,94 | 66,65 | 59,73 | 41,13 | 47,84 | 0,906 |

Điểm được chấm lại với cùng quy tắc: NFKC, casefold, gộp khoảng trắng, bỏ hai
token cấu trúc `##LTLine##`, `##LTFigure##`. CER/WER có thể vượt 100%.
Xem [phân tích theo nhóm](latex_report/analysis/common_test_by_category.csv) và
[biểu đồ theo loại che](latex_report/img/robustness.pdf).

## Tệp kết quả và tái tạo

- `result/`: CSV gốc (nén `.csv.gz` trên Git), cấu hình, bảng nhóm và hình export.
- `latex_report/analysis/`: điểm chấm lại, bảng dùng trong báo cáo, ba ví dụ có ID
  và GT/dự đoán đầy đủ, cùng hash các đầu vào đã kiểm tra.
- `paddleocr_isolated_benchmark.py`, `tesseract_isolated_benchmark.py`: worker
  standalone; notebook đã nhúng worker nên không cần upload riêng.

Dùng Python 3.11 trở lên với môi trường ảo:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python scripts/results_archive.py restore
python latex_report/analysis/analyze_results.py
python latex_report/analysis/build_report_assets.py
```

Các CSV khôi phục được kiểm tra SHA-256. Đọc GT bằng Python CSV engine để giữ
ký tự NUL; không dùng C engine mặc định để chấm lại. Để cập nhật archive sau khi
có export mới: `python scripts/results_archive.py pack`.

## Notebook và chạy tiếp

Mỗi notebook có bước cài đặt và cấu hình riêng. M1–M4 cần GPU CUDA; Tesseract
chạy CPU. Chọn cả hai nhóm và `MAX_SAMPLES_PER_CATEGORY = None` để chạy toàn test.
Ảnh dataset và trọng số được tải khi chạy, không đưa vào Git.

[Notebook Resume](OPRB_Florence_2_Base_Resume.ipynb) tự khôi phục CSV/CSV.gz và
`checkpoint_config.json` từ output Florence đã gắn qua Add Input. Nó giữ tiến độ
trong thư mục làm việc và chỉ thử ảnh lỗi/chưa thành công. Bản này đã sửa lỗi
ảnh 1×1 và lỗi trộn chuỗi/số khi tổng hợp sau resume.

Nếu đã có đủ kết quả hiện tại, không cần suy luận lại để tái tạo báo cáo.
Notebook Qwen, thư mục thử nghiệm `test/`, tệp build LaTeX tạm và nội dung không
liên quan của template cũ được giữ ở máy cá nhân, không đưa lên Git.
