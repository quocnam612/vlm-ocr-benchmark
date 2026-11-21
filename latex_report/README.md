# Báo cáo Lab 2 — đánh giá VLM và OCR trên OPRB

`main.pdf` là bản báo cáo tiếng Việt, **3 trang A4 gồm cả bìa**, theo `../LabOCR.pdf`.
Giữ template HCMUS, logo, header/footer, hai giảng viên và ngày bìa `\today`.
Có đủ năm mục: Giới thiệu, Thống kê dữ liệu, Mô tả mô hình, Kết quả thực nghiệm,
Kết luận. Tiêu đề không còn phần chú thích tiếng Anh.
Tên năm mô hình màu xanh trong bảng ở Mục 3 liên kết đến thư mục kết quả chạy
trên Google Drive do tác giả cung cấp. Các phần liệt kê dữ liệu, cách chấm,
nhận xét kết quả và ví dụ được trình bày bằng bullet.

## Dữ liệu đã kiểm tra

Năm mô hình đều đã duyệt 9.000 ảnh. Florence/EasyOCR/docTR có 9.000 dòng `ok`;
PaddleOCR/Tesseract có 8.999 dòng `ok` và cùng một dòng lỗi chấm điểm vì GT chỉ
có `##LTFigure##`. Loại ảnh đó nhất quán, so sánh cả năm trên **cùng 8.999 ảnh**
(6.000 text, 2.999 nontext). Không còn dùng tập con Florence 5.790 ảnh.

`analysis/analyze_results.py` đã tính lại và đối chiếu 44.998 bản ghi thành công
với điểm gốc, bảng nhóm, cấu hình và GT. Bảng báo cáo dùng NFKC, casefold, gộp
khoảng trắng, bỏ đúng hai token `##LTLine##`, `##LTFigure##`, tính từng ảnh rồi
trung bình đều. CER/WER không bị cắt về 100%.

Các tệp chính:

- `analysis/summary_common_test.csv`: CER/WER/P/R/F1/thời gian của năm mô hình.
- `analysis/common_test_by_category.csv`, `common_test_by_occlusion_type.csv`:
  kết quả theo vùng và loại che.
- `analysis/common_test_ids.csv`: danh sách 8.999 ảnh chung.
- `analysis/coverage.csv`, `execution_errors.csv`, `excluded_references.csv`:
  số ảnh, lỗi gốc và GT bị loại.
- `analysis/audit_summary.json`: kết quả kiểm tra và SHA-256 tệp đầu vào.
- `analysis/qualitative_examples.json`: ID đầy đủ, GT/dự đoán và điểm của ba ví dụ.
- `analysis/generated/`: hai bảng LaTeX; `img/robustness.pdf`: biểu đồ vector.

CSV phải đọc bằng `engine="python"` và tăng `csv.field_size_limit`, để giữ
các ký tự NUL và GT dài. Mật độ chỉ nhận đúng `0p5pct`, `1p0pct`, `2p0pct`;
trường thứ hai trong tên Sim/Through_Stamp là ID archive, không phải mật độ.
Các bảng mật độ gốc được giữ nguyên; phân tích đã sửa cách phân nhóm.

## Tái tạo

Từ thư mục gốc repo, cài `requirements-analysis.txt`. Nếu lấy dữ liệu từ Git,
chạy `python scripts/results_archive.py restore` trước. Sau đó:

```sh
python latex_report/analysis/analyze_results.py
python latex_report/analysis/build_report_assets.py
```

Biên dịch trong `latex_report/`:

```sh
sh build.sh
pdfinfo main.pdf
```

`build.sh` gọi `latexmk`, tự chạy BibTeX và đủ các lượt LaTeX. Nếu chỉ chạy một
lượt pdfLaTeX khi chưa có tệp phụ, các trích dẫn có thể hiện `[?]`. Tệp chính
đã đặt công cụ biên dịch là `latexmk`; bản PDF giao kèm đã được kiểm tra không
còn dấu hỏi ở trích dẫn.

Trên Overleaf chọn `main.tex` và pdfLaTeX. Hai bảng và hình đã có sẵn nên biên
dịch PDF không cần chạy Python. Thư mục `code/`, `member.tex`, `phuluc.tex` là
nội dung bài cũ, không được nhập vào báo cáo này.
