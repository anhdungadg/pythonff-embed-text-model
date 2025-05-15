# Titan Embed Text

Công cụ tạo embedding vector từ văn bản sử dụng Amazon Bedrock Titan Embeddings.

## Mô tả

Chương trình Python này cung cấp một lớp đơn giản để tạo embedding vector từ văn bản sử dụng mô hình Amazon Titan Embed Text thông qua dịch vụ Amazon Bedrock. Embedding vector là biểu diễn số học của văn bản, cho phép so sánh ngữ nghĩa giữa các đoạn văn bản khác nhau.

## Tính năng

- Tạo embedding vector từ văn bản sử dụng Amazon Titan Embed Text
- Tùy chỉnh số chiều của vector embedding
- Tùy chọn chuẩn hóa vector embedding
- Dễ dàng tích hợp vào các ứng dụng xử lý ngôn ngữ tự nhiên

## Yêu cầu

- Python 3.x
- Boto3 (AWS SDK cho Python)
- Tài khoản AWS với quyền truy cập Amazon Bedrock

Cài đặt thư viện cần thiết:

```bash
pip install boto3
```

## Cấu hình AWS

Đảm bảo bạn đã cấu hình thông tin xác thực AWS trên máy tính của mình. Bạn có thể sử dụng AWS CLI để cấu hình:

```bash
aws configure
```

## Cách sử dụng

### Khởi tạo lớp TitanEmbeddings

```python
from titan_embed_text import TitanEmbeddings

# Sử dụng mô hình mặc định (amazon.titan-embed-text-v2:0)
titan_embeddings = TitanEmbeddings()

# Hoặc chỉ định một mô hình khác
titan_embeddings = TitanEmbeddings(model_id="amazon.titan-embed-text-v1")
```

### Tạo embedding cho văn bản

```python
# Tạo embedding với các tham số mặc định (1024 chiều, có chuẩn hóa)
input_text = "Xin hãy giới thiệu sách có chủ đề tương tự phim Inception."
embedding = titan_embeddings(input_text)

# Tùy chỉnh số chiều và chuẩn hóa
embedding = titan_embeddings(input_text, dimensions=768, normalize=True)

# In ra 10 phần tử đầu tiên của embedding
print(f"10 phần tử đầu tiên của embedding: {embedding[:10]}")
```

## Tham số

### Khởi tạo TitanEmbeddings

- `model_id` (str, tùy chọn): ID của mô hình Titan Embeddings. Mặc định là "amazon.titan-embed-text-v2:0".

### Phương thức __call__

- `text` (str): Văn bản cần tạo embedding.
- `dimensions` (int, tùy chọn): Số chiều của vector embedding. Mặc định là 1024.
- `normalize` (bool, tùy chọn): Có chuẩn hóa vector embedding hay không. Mặc định là True.

## Giá trị trả về

- `List[float]`: Vector embedding của văn bản đầu vào.

## Ứng dụng

- Tìm kiếm ngữ nghĩa (semantic search)
- Phân loại văn bản
- Phát hiện văn bản tương tự
- Hệ thống gợi ý
- Phân tích cảm xúc
- Tích hợp với các mô hình ngôn ngữ lớn (LLM)

## Lưu ý

- Đảm bảo tài khoản AWS của bạn có quyền truy cập vào dịch vụ Amazon Bedrock và mô hình Titan Embeddings.
- Việc sử dụng dịch vụ Amazon Bedrock có thể phát sinh chi phí theo mức giá của AWS.
