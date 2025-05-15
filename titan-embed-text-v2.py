import json
import boto3
import time
import concurrent.futures
from typing import List, Optional, Dict, Union, Tuple


class TitanEmbeddings:
    accept = "application/json"
    content_type = "application/json"
    
    def __init__(self, model_id="amazon.titan-embed-text-v2:0", use_cache=True, region_name=None):
        """
        Khởi tạo lớp TitanEmbeddings
        
        Args:
            model_id (str): ID của mô hình Titan Embeddings
            use_cache (bool): Có sử dụng cache để lưu trữ kết quả hay không
            region_name (str, optional): Tên region AWS, nếu None sẽ sử dụng region mặc định
        """
        self.bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.use_cache = use_cache
        self.cache = {}
    
    def __call__(self, text: str, dimensions: int = 1024, normalize: bool = True, 
                 measure_performance: bool = False) -> Optional[List[float]]:
        """
        Trả về Titan Embeddings
        
        Args:
            text (str): văn bản cần tạo embedding
            dimensions (int): Số chiều đầu ra
            normalize (bool): Chuẩn hóa embedding hay không
            measure_performance (bool): Có đo hiệu suất hay không
            
        Return:
            List[float]: Embedding vector hoặc None nếu có lỗi
        """
        # Kiểm tra cache
        cache_key = f"{text}_{dimensions}_{normalize}"
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Đo hiệu suất nếu được yêu cầu
        start_time = time.time() if measure_performance else None
        
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": dimensions,
                "normalize": normalize
            })
            
            response = self.bedrock.invoke_model(
                body=body, 
                modelId=self.model_id, 
                accept=self.accept, 
                contentType=self.content_type
            )
            
            response_body = json.loads(response.get('body').read())
            embedding = response_body['embedding']
            
            # Lưu vào cache nếu được yêu cầu
            if self.use_cache:
                self.cache[cache_key] = embedding
                
            # Hiển thị thông tin hiệu suất nếu được yêu cầu
            if measure_performance:
                elapsed_time = time.time() - start_time
                print(f"Thời gian tạo embedding: {elapsed_time:.4f} giây")
                
            return embedding
        except Exception as e:
            print(f"Lỗi khi tạo embedding: {str(e)}")
            return None
    
    def batch_embed(self, texts: List[str], dimensions: int = 1024, 
                    normalize: bool = True) -> List[Optional[List[float]]]:
        """
        Tạo embeddings cho nhiều văn bản cùng lúc
        
        Args:
            texts (List[str]): Danh sách văn bản cần tạo embedding
            dimensions (int): Số chiều đầu ra
            normalize (bool): Chuẩn hóa embedding hay không
            
        Return:
            List[List[float]]: Danh sách các embedding vector
        """
        results = []
        for text in texts:
            results.append(self(text, dimensions, normalize))
        return results
    
    def parallel_batch_embed(self, texts: List[str], dimensions: int = 1024, 
                            normalize: bool = True, max_workers: int = 4) -> List[Optional[List[float]]]:
        """
        Tạo embeddings cho nhiều văn bản cùng lúc sử dụng đa luồng
        
        Args:
            texts (List[str]): Danh sách văn bản cần tạo embedding
            dimensions (int): Số chiều đầu ra
            normalize (bool): Chuẩn hóa embedding hay không
            max_workers (int): Số luồng tối đa
            
        Return:
            List[List[float]]: Danh sách các embedding vector
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self, text, dimensions, normalize) for text in texts]
            return [future.result() for future in concurrent.futures.as_completed(futures)]
    
    def clear_cache(self) -> None:
        """
        Xóa cache
        """
        self.cache = {}
    
    def save_embedding(self, embedding: List[float], file_path: str) -> None:
        """
        Lưu embedding vào file
        
        Args:
            embedding (List[float]): Embedding vector cần lưu
            file_path (str): Đường dẫn file
        """
        with open(file_path, 'w') as f:
            json.dump(embedding, f)
    
    def load_embedding(self, file_path: str) -> List[float]:
        """
        Tải embedding từ file
        
        Args:
            file_path (str): Đường dẫn file
            
        Return:
            List[float]: Embedding vector
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Tính độ tương đồng cosine giữa hai embedding
        
        Args:
            embedding1 (List[float]): Embedding vector thứ nhất
            embedding2 (List[float]): Embedding vector thứ hai
            
        Return:
            float: Độ tương đồng cosine (từ -1 đến 1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Hai embedding phải có cùng kích thước")
            
        dot_product = sum(a*b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a*a for a in embedding1) ** 0.5
        norm2 = sum(b*b for b in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def visualize_embeddings(self, embeddings: List[List[float]], labels: Optional[List[str]] = None) -> None:
        """
        Trực quan hóa các embedding sử dụng t-SNE
        
        Args:
            embeddings (List[List[float]]): Danh sách các embedding vector
            labels (List[str], optional): Nhãn cho các embedding
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
            
            if labels:
                for i, label in enumerate(labels):
                    plt.annotate(label, (reduced_data[i, 0], reduced_data[i, 1]))
                    
            plt.title("t-SNE visualization of embeddings")
            plt.show()
        except ImportError:
            print("Cần cài đặt scikit-learn và matplotlib để sử dụng tính năng này")
    
    @staticmethod
    def get_model_for_language(language: str) -> str:
        """
        Trả về model_id phù hợp với ngôn ngữ
        
        Args:
            language (str): Mã ngôn ngữ (ví dụ: 'vi', 'en')
            
        Return:
            str: Model ID phù hợp
        """
        language_map = {
            "vi": "amazon.titan-embed-text-v2:0",  # Tốt cho tiếng Việt
            "en": "amazon.titan-embed-text-v2:0",  # Tiếng Anh
            "zh": "amazon.titan-embed-text-v2:0",  # Tiếng Trung
            "ja": "amazon.titan-embed-text-v2:0",  # Tiếng Nhật
            "ko": "amazon.titan-embed-text-v2:0",  # Tiếng Hàn
            # Thêm các ngôn ngữ khác
        }
        return language_map.get(language, "amazon.titan-embed-text-v2:0")  # Mặc định
    
    def find_similar_texts(self, query_text: str, texts: List[str], 
                          top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Tìm các văn bản tương tự với văn bản truy vấn
        
        Args:
            query_text (str): Văn bản truy vấn
            texts (List[str]): Danh sách các văn bản cần so sánh
            top_k (int): Số lượng kết quả trả về
            
        Return:
            List[Tuple[int, float, str]]: Danh sách (index, độ tương đồng, văn bản) được sắp xếp theo độ tương đồng giảm dần
        """
        query_embedding = self(query_text)
        if query_embedding is None:
            return []
            
        similarities = []
        for i, text in enumerate(texts):
            text_embedding = self(text)
            if text_embedding is not None:
                similarity = self.cosine_similarity(query_embedding, text_embedding)
                similarities.append((i, similarity, text))
                
        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Trả về top_k kết quả
        return similarities[:top_k]


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo với cache
    titan_embeddings_v2 = TitanEmbeddings(use_cache=True)
    
    # Tạo embedding đơn lẻ
    input_text = "Xin hãy giới thiệu sách có chủ đề tương tự phim Inception."
    embedding = titan_embeddings_v2(input_text, dimensions=1024, normalize=True, measure_performance=True)
    print(f"10 phần tử đầu tiên của embedding: {embedding[:10]}")
    
    # Tạo embedding cho nhiều văn bản
    texts = [
        "Trí tuệ nhân tạo đang phát triển nhanh chóng.",
        "Machine learning là một lĩnh vực của AI.",
        "Deep learning là một kỹ thuật trong machine learning."
    ]
    
    # Sử dụng xử lý tuần tự
    print("\nXử lý tuần tự:")
    embeddings = titan_embeddings_v2.batch_embed(texts)
    
    # Sử dụng xử lý song song
    print("\nXử lý song song:")
    parallel_embeddings = titan_embeddings_v2.parallel_batch_embed(texts, max_workers=3)
    
    # Tính độ tương đồng
    similarity = TitanEmbeddings.cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nĐộ tương đồng giữa văn bản 1 và 2: {similarity:.4f}")
    
    # Tìm văn bản tương tự
    query = "AI và machine learning có mối quan hệ như thế nào?"
    similar_texts = titan_embeddings_v2.find_similar_texts(query, texts)
    print("\nCác văn bản tương tự với truy vấn:")
    for idx, score, text in similar_texts:
        print(f"- {text} (Độ tương đồng: {score:.4f})")
    
    # Trực quan hóa (cần cài đặt scikit-learn và matplotlib)
    # titan_embeddings_v2.visualize_embeddings(embeddings, labels=texts)
