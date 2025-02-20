import pickle
import numpy as np
import faiss
import torch
import clip
from PIL import Image

class ImageHandler:
    def __init__(self, image_index_path, product_urls_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.image_index = faiss.read_index(image_index_path)
        with open(product_urls_path, 'rb') as f:
            self.product_urls = pickle.load(f)

    def get_image_embedding(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(img_tensor).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None

    def find_similar_images(self, query_image_path, num_results=3):
        query_embedding = self.get_image_embedding(query_image_path)
        if query_embedding is None:
            return []

        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.image_index.search(query_embedding, num_results)

        return [
            {'url': self.product_urls[idx], 'similarity_score': 1 - (distance / 2)}
            for idx, distance in zip(indices[0], distances[0]) if idx < len(self.product_urls)
        ]
#
# if __name__ == "__main__":
#     image = ImageHandler('image_index/image_faiss.index',
#                          'image_index/products_url.pkl')
#     image_path = 'test_img.png'
#     if image_path:
#         similar_products = image.find_similar_images(image_path)
#         if similar_products:
#             print("Here are similar products I found:")
#             for idx, product in enumerate(similar_products, 1):
#                 similarity_percentage = round(product['similarity_score'] * 100, 2)
#                 print(
#                     f"{idx}. {product['url']} (Similarity: {similarity_percentage}%)"
#                 )
#         else:
#             print("I couldn't find any similar products for the provided image.")