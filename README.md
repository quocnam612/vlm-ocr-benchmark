# VLM / OCR Benchmark trên OPRB

Bài Lab 2: so sánh năm mô hình pretrained nhẹ trên ảnh tài liệu bị che khuất
của [OPRB](https://huggingface.co/datasets/kpurkayastha/OPRB), chỉ dùng tập test.
Không huấn luyện hoặc tinh chỉnh trên test; GT chỉ dùng để chấm điểm.

**Báo cáo tiếng Việt:** [PDF, 3 trang A4](latex_report/main.pdf) ·
[Mã LaTeX và hướng dẫn](latex_report/README.md) · [Yêu cầu bài lab](LabOCR.pdf).

## Mô hình và phạm vi hiện tại

| Mã | Mô hình / pipeline | Notebook | Số ảnh có điểm hợp lệ |
| --- | --- | --- | ---: |
| M1 | Florence-2-base, FP16 | [Florence](OPRB_Florence_2_Base_Standard_Decoding.ipynb) | 5.790, đều thuộc text |
| M2 | EasyOCR: CRAFT + english_g2 | [EasyOCR](OPRB_EasyOCR_English_Test_Evaluation.ipynb) | 8.999 |
| M3 | docTR: DB / CRNN MobileNetV3 | [docTR](OPRB_docTR_MobileNet_Test_Evaluation.ipynb) | 8.999 |
| M4 | PP-OCRv5 mobile detector + English recognizer | [PaddleOCR](OPRB_PaddleOCR_PP_OCRv5_Mobile_English_Test_Evaluation.ipynb) | 8.999 |
| M5 | Tesseract 5, English tessdata_fast | [Tesseract](OPRB_Tesseract_English_Fast_Test_Evaluation.ipynb) | 8.999 |

Test có 9.000 ảnh: 6.000 text và 3.000 nontext. Bốn OCR đã duyệt đủ 9.000 ảnh;
loại chung một GT chỉ chứa marker cấu trúc, còn 8.999 ảnh chấm được.
Florence chủ động dừng ở ngân sách **10 giờ/phiên**, đã thành công 5.790 ảnh,
có 2 lỗi và 3.208 ảnh chưa thử: **còn 3.210 ảnh chưa có kết quả thành công**.
Repo và báo cáo hiện chưa hoàn tất toàn bộ test cho cả năm mô hình.

Qwen đã được loại khỏi benchmark hiện tại. Các lần thử nhỏ và nội dung không
liên quan của bài LaTeX cũ được giữ ở máy cá nhân, không đưa vào repo.

## Kết quả so sánh trên cùng 5.790 ảnh text

Các chỉ số là trung bình macro theo ảnh, dùng %, thời gian dùng giây/ảnh.
M1–M4 chạy trên một Tesla T4; M5 chạy CPU Xeon, nên thời gian CPU/GPU không phải
phép so sánh trên cùng phần cứng.

| Mô hình | CER ↓ | WER ↓ | Precision ↑ | Recall ↑ | F1 ↑ | Giây/ảnh ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence | 56,04 | 77,55 | 53,67 | 31,63 | 38,70 | 5,698 |
| EasyOCR | 55,39 | 81,05 | 44,56 | 31,96 | 36,76 | 1,351 |
| docTR | 43,85 | 59,76 | 78,38 | 66,18 | 71,14 | 0,595 |
| PaddleOCR | **24,66** | **38,08** | **85,62** | **70,78** | **76,75** | **0,389** |
| Tesseract | 43,82 | 63,76 | 60,42 | 42,40 | 49,05 | 0,940 |

Trên 8.999 ảnh GT hợp lệ, PaddleOCR đạt CER 33,97%, WER 43,25%, F1 77,75%.
Xem [bảng theo nhóm](latex_report/analysis/common_ocr_by_category.csv) và
[biểu đồ độ bền vững](latex_report/img/robustness.pdf).

Quy tắc chung: NFKC, casefold, gộp khoảng trắng và loại đúng hai token
`##LTLine##`, `##LTFigure##` ở GT lẫn dự đoán. Vì các export cũ chưa cùng quy tắc
marker, báo cáo dùng điểm chấm lại trong `latex_report/analysis/`, không trộn
trực tiếp các bảng gốc. CER/WER có thể vượt 100%; không cắt giá trị ngoại lệ.

## Nội dung repo

```text
OPRB_*.ipynb                 Notebook chạy độc lập cho từng mô hình
paddleocr_isolated_benchmark.py
tesseract_isolated_benchmark.py  Bản standalone của worker nhúng trong notebook
result/                     Export gốc: cấu hình, bảng nhóm, hình và CSV nén
latex_report/               Template HCMUS, báo cáo PDF/LaTeX, phân tích dẫn xuất
scripts/results_archive.py  Nén/khôi phục CSV, kiểm tra SHA-256
requirements-analysis.txt  Phụ thuộc để tái tạo phân tích và biểu đồ
```

Không đưa ảnh dataset, trọng số mô hình hay tệp build LaTeX tạm vào Git.
Các file `per_image_results.csv.gz` là **bản nén không mất dữ liệu** của CSV gốc;
Git không theo dõi bản `.csv` đã giải nén. Các hash đầu vào được lưu trong
[`audit_summary.json`](latex_report/analysis/audit_summary.json).

## Chạy notebook trên Kaggle

Mỗi notebook chứa bước cài đặt và cấu hình riêng. M1–M4 cần GPU CUDA;
Tesseract chạy CPU. Notebook tải ảnh test và trọng số pretrained khi chạy.
PaddleOCR/Tesseract dùng môi trường tách riêng được thiết lập trong notebook;
không cần upload hai worker `.py` riêng nếu chạy notebook.

Thử cấu hình mẫu nhỏ trước; để đánh giá toàn bộ test, đặt
`MAX_SAMPLES_PER_CATEGORY = None` và chọn cả `text`, `nontext`.
Giữ thư mục kết quả riêng cho từng mô hình; xuất và lưu output sau mỗi phiên.

### Chạy tiếp Florence

1. Gắn output Florence của phiên trước vào phiên mới.
2. Khôi phục **cả** `per_image_results.csv` và `checkpoint_config.json` vào
   `/kaggle/working/oprb_florence_standard_results/` trước cell vòng lặp.
3. Giữ nguyên cấu hình mô hình/giải mã, chọn cả hai nhóm, đặt
   `MAX_SAMPLES_PER_CATEGORY = None` và `MAX_EVAL_HOURS = 10.0`.
4. Kiểm tra log `Reusing 5,790 successful selected pages; 3,210 remaining.`
   trước khi để notebook chạy tiếp. Hai ảnh lỗi cũ sẽ được thử lại.

Nếu lấy checkpoint từ repo này, giải nén CSV trước khi upload; file JSON nằm
cùng thư mục kết quả. Không bỏ kiểm tra cấu hình khi môi trường khác checkpoint.

## Tái tạo phân tích và báo cáo

Cần Python 3.11 trở lên và môi trường ảo riêng:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python scripts/results_archive.py restore
python latex_report/analysis/analyze_results.py
python latex_report/analysis/build_report_assets.py
```

Quy trình đã đối chiếu 41.788 bản ghi `status=ok` với điểm gốc và các bảng tổng.
Đọc CSV bằng Python engine để giữ các ký tự NUL trong GT; không dùng C engine
mặc định để chấm lại. Script hiện kiểm tra chặt các cohort 5.790 / 8.999;
khi cập nhật kết quả Florence cần cập nhật cohort và các nhận xét tương ứng.

Biên dịch LaTeX bằng pdfLaTeX/BibTeX (cần TeX Live có hỗ trợ tiếng Việt):

```sh
cd latex_report
TZ=Asia/Ho_Chi_Minh latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Đã có sẵn bảng LaTeX và hình vector để biên dịch PDF mà không chạy lại phân tích.
Giữ template HCMUS, logo, bìa và hai giảng viên; ngày bìa dùng `\today`, cập nhật
khi biên dịch. Báo cáo giới hạn 3 trang A4 kể cả bìa.

Sau khi có export mới, kiểm tra/chấm lại rồi chạy
`python scripts/results_archive.py pack` để cập nhật các archive trước khi commit.
