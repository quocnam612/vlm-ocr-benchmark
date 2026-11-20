# Báo cáo Lab 2 — OPRB, VLM và OCR

`main.pdf` là bản báo cáo tiếng Việt, dùng **template HCMUS có sẵn**: giữ
`hcmus-report-template.sty`, bìa `content/title.tex`, logo, header/footer và BibTeX.
Báo cáo gồm bìa và hai trang nội dung A4, đủ năm mục theo `../LabOCR.pdf`.
Giảng viên: PGS.TS. Đinh Điền và TS. Nguyễn Hồng Bửu Long. Ngày bìa dùng
`\today`, cập nhật khi biên dịch lại (ngày trong PDF không tự đổi nếu không biên dịch).

## Kết quả và phạm vi

- M1 Florence-2-base; M2 EasyOCR; M3 docTR MobileNet; M4 PaddleOCR PP-OCRv5
  mobile English; M5 Tesseract English tessdata_fast. Qwen không nằm trong báo cáo.
- Kết quả gốc nằm trong `../result/`, không bị sửa bởi quy trình phân tích.
  Trên Git, CSV được lưu dạng `.csv.gz`; sau khi clone, chạy
  `python ../scripts/results_archive.py restore` trong thư mục `latex_report/`
  để khôi phục CSV trước khi chạy phân tích.
- Florence: 5.792 ảnh đã thử, 5.790 thành công (đều là text), 2 lỗi chuẩn hóa
  số kênh, 3.208 ảnh chưa thử. Tổng cộng **3.210 ảnh chưa có kết quả thành công**.
  Log xác nhận notebook dừng chủ động vì `MAX_EVAL_HOURS = 10`, không phải idle.
- Bốn OCR đã duyệt đủ 9.000 ảnh. GT của một ảnh nontext chỉ có nhãn cấu trúc
  `##LTFigure##`, không có văn bản tham chiếu. Paddle/Tesseract ghi ảnh này là
  lỗi chấm điểm; EasyOCR/docTR ghi ok theo quy tắc cũ. Loại cùng ảnh khỏi cả bốn
  khi phân tích, còn **8.999 ảnh hợp lệ** (6.000 text + 2.999 nontext).
- Bảng chính so sánh năm mô hình trên **cùng 5.790 ảnh text**, không so sánh
  trung bình của các tập khác nhau. Bảng tiếp theo đánh giá bốn OCR trên cùng 8.999 ảnh.
- Báo cáo ghi rõ Florence chưa hoàn tất toàn test, chưa đáp ứng trọn vẹn phạm vi
  toàn bộ test cho cả năm mô hình. Không coi việc dùng tập chung là đã được
  giảng viên chấp thuận thay thế yêu cầu đó.

## Kiểm tra và chấm lại

`analysis/analyze_results.py` đã kiểm tra 41.788 bản ghi `status=ok`: tính lại
CER/WER/P/R/F1 và số từ, đối chiếu điểm từng ảnh, bảng tổng, bảng theo nhóm,
GT giữa các mô hình, cấu hình, checkpoint, ảnh trùng và thời gian dương.
`analysis/audit_summary.json` lưu SHA-256 các CSV/cấu hình đầu vào và kết quả kiểm tra.

Các bảng báo cáo được tính lại nhất quán bằng NFKC, casefold, gộp khoảng trắng,
loại đúng token `##LTLine##`, `##LTFigure##` ở cả GT và dự đoán. Điểm macro là
trung bình đều theo ảnh; không cắt CER/WER tại 100%. Bảng gốc Florence/EasyOCR/docTR
chưa loại marker nên có giá trị khác; không trộn chính sách chấm giữa mô hình.

Đọc CSV bằng **`engine="python"`**, tăng `csv.field_size_limit`: một số GT có ký
tự NUL và trường rất dài. Engine C mặc định có thể cắt chuỗi tại NUL, làm điểm
chấm lại sai. Giữ nguyên các ký tự còn lại theo quy tắc notebook.

Không suy mật độ từ trường thứ hai của tên ảnh nếu không khớp đúng
`0p5pct`, `1p0pct`, `2p0pct`. Với Sim/Through_Stamp, trường đó là ID archive,
không phải mật độ. Có 2.852 ảnh mật độ chưa xác định, vẫn tính trong bảng tổng
và theo loại; chỉ bỏ khỏi biểu đồ ba mức mật độ (6.147 ảnh GT hợp lệ).

Các GT rất ngắn còn gây ngoại lệ CER/WER lớn (ví dụ E3 có CER = 510 lần).
Đây là số thực tế theo công thức, không phải tỷ lệ đã cắt về [0, 1]. Báo cáo
nêu giới hạn nhãn, không xem mọi ký tự trong hình mà OCR đọc được là lỗi mô hình.

Dữ liệu dẫn xuất:

- `analysis/coverage.csv`, `execution_errors.csv`, `excluded_references.csv`:
  phạm vi, lỗi và GT không hợp lệ.
- `analysis/scored/M*.csv`: điểm chấm lại từng ảnh, không nhân bản toàn bộ GT/dự đoán.
- `analysis/summary_common_text.csv`, `summary_common_ocr.csv` và `*_by_*.csv`:
  các bảng theo tập chung; `*_ids.csv` lưu danh sách ảnh chính xác.
- `analysis/qualitative_examples.json`: ID đầy đủ, GT/dự đoán và điểm của E1–E3;
  các đoạn trích trong PDF không thay cho toàn bộ nội dung ảnh.
- `analysis/build_report_assets.py`: tạo bảng LaTeX và `img/robustness.pdf` (vector),
  kèm PNG xem trước từ dữ liệu dẫn xuất.

Tái tạo (cần Python có pandas, rapidfuzz, matplotlib):

```sh
python analysis/analyze_results.py
python analysis/build_report_assets.py
```

Không cần chạy lại suy luận để tái tạo báo cáo. Khi có export Florence đầy đủ,
cập nhật đúng thư mục `../result/oprb_florence_standard_results/`, rồi cập nhật
phạm vi/cohort trong script và nội dung nhận xét trước khi biên dịch. Script hiện
cố ý kiểm tra số lượng 5.790 và 8.999 để tránh dùng nhầm export.

## Chạy tiếp Florence trên Kaggle

Lưu lại output của phiên cũ. Ở phiên mới, khôi phục **cả** `per_image_results.csv`
và `checkpoint_config.json` vào `OUTPUT_DIR` mà notebook đang dùng
(`/kaggle/working/oprb_florence_standard_results`). Nếu gắn output phiên trước
qua Add Input, phải sao chép từ đường dẫn read-only `/kaggle/input/...` sang
`OUTPUT_DIR` trước cell vòng lặp. Giữ nguyên cấu hình mô hình/giải mã, chọn cả hai
nhóm và `MAX_SAMPLES_PER_CATEGORY = None`. Kiểm tra log báo tái sử dụng 5.790
trang thành công rồi chạy tiếp; hai ảnh lỗi sẽ được thử lại. Tăng riêng ngân sách
thời gian không sửa lỗi chuẩn hóa của hai ảnh đó.

## Biên dịch

Chạy trong `latex_report/`:

```sh
TZ=Asia/Ho_Chi_Minh latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
```

Hoặc dùng pdfLaTeX → BibTeX → pdfLaTeX hai lần. Trên Overleaf chọn `main.tex`
và pdfLaTeX. Các tệp `analysis/generated/*_table.tex` và `img/robustness.pdf` đã có
sẵn nên chỉ biên dịch không cần Python hay dữ liệu `result/`.
`code/`, `member.tex`, `phuluc.tex` thuộc template/bài cũ, không được nhập vào PDF.
