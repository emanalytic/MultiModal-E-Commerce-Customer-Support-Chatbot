import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
import pickle
import numpy as np
import faiss
import torch
import clip
from PIL import Image

class MultimodalShoppingAssistant:
    def __init__(self, vectorDB_path, image_index_path, product_urls_path):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.vectorDB = LangchainFAISS.load_local(
            vectorDB_path,
            HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
            allow_dangerous_deserialization=True
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.image_index = faiss.read_index(image_index_path)

        with open(product_urls_path, 'rb') as f:
            self.product_urls = pickle.load(f)

        self.conversation_chain, self.memory = self.get_conversation_chain()

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)

    def get_conversation_chain(self):
        llm = ChatGroq(
            api_key='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',
            model_name='llama3-70b-8192',
            temperature=0
        )
        with open('prompt.txt', 'r') as file:
            prompt_text = file.read().strip()

        prompt_chatbot = ChatPromptTemplate.from_messages([
            ('system', prompt_text),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        context_retriever = create_history_aware_retriever(
            llm,
            self.vectorDB.as_retriever(),
            prompt_chatbot
        )

        document_chain = create_stuff_documents_chain(llm, prompt_chatbot)
        conversation_chain = create_retrieval_chain(context_retriever, document_chain)
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key='result',
            input_key='input'
        )

        return conversation_chain, memory

    def get_image_embedding(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(img_tensor)
                embedding = embedding.cpu().numpy().flatten()
            return embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None

    def find_similar_images(self, query_image_path, num_results=3):
        try:
            query_embedding = self.get_image_embedding(query_image_path)
            if query_embedding is None:
                return []

            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
            distances, indices = self.image_index.search(query_embedding, num_results)

            similar_products = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.product_urls):
                    similar_products.append({
                        'url': self.product_urls[idx],
                        'similarity_score': 1 - (distance / 2)
                    })

            return similar_products
        except Exception as e:
            print(f"Error finding similar images: {e}")
            return []

    def process_query(self, user_input, image_path=None):
        if not user_input and not image_path:
            return "I'm sorry, I couldn't process that. Could you please try again?"

        response_parts = []

        if image_path:
            similar_products = self.find_similar_images(image_path)
            if similar_products:
                response_parts.append("Here are similar products I found:")
                for idx, product in enumerate(similar_products, 1):
                    similarity_percentage = round(product['similarity_score'] * 100, 2)
                    response_parts.append(f"{idx}. {product['url']} (Similarity: {similarity_percentage}%)")
            else:
                response_parts.append("I couldn't find any similar products for the provided image.")

        if user_input:
            documents_retrieved = self.vectorDB.similarity_search(user_input, threshold=0.9)
            if not documents_retrieved:
                response_parts.append("I don't have any information on that particular product. Could you please specify another product or feature you're interested in?")
            else:
                context = " ".join([doc.page_content for doc in documents_retrieved if hasattr(doc, 'page_content')])
                chain_input = {
                    'input': user_input,
                    'chat_history': self.memory.chat_memory.messages,
                    'context': context
                }
                result = self.conversation_chain.invoke(chain_input)
                response_parts.append(result['answer'])
                self.memory.save_context({'input': user_input}, {'result': result['answer']})

        return "\n\n".join(response_parts)
