import json
import boto3

class TitanEmbeddings:
    accept = "application/json"
    content_type = "application/json"
    
    def __init__(self, model_id="amazon.titan-embed-text-v2:0"):
        self.bedrock = boto3.client(service_name='bedrock-runtime')
        self.model_id = model_id
    
    def __call__(self, text, dimensions=1024, normalize=True):
        """
        Trả về Titan Embeddings
        
        Args:
            text (str): văn bản cần tạo embedding
            dimensions (int): Số chiều đầu ra
            normalize (bool): Chuẩn hóa embedding hay không
            
        Return:
            List[float]: Embedding vector
        """
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
        return response_body['embedding']

# Sử dụng
titan_embeddings_v2 = TitanEmbeddings()
input_text = "Xin hãy giới thiệu sách có chủ đề tương tự phim Inception."
embedding = titan_embeddings_v2(input_text, dimensions=1024, normalize=True)
print(f"10 phần tử đầu tiên của embedding: {embedding[:10]}")
