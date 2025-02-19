import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
import time
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

    def get_voice_input(self):
        try:
            with self.microphone as source:
                print("\nListening... (Speak now)")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)

            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text

        except sr.WaitTimeoutError:
            print("No speech detected within timeout period.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

    def get_text_input(self):
        try:
            user_input = input("Enter your query: ")
            return user_input.strip()
        except Exception as e:
            print(f"Error reading text input: {e}")
            return None

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
        """Find similar images using FAISS and CLIP embeddings"""
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
        """Process user query based on input type"""
        if not user_input and not image_path:
            return "I'm sorry, I couldn't process that. Could you please try again?"

        response_parts = []

        if image_path:
            similar_products = self.find_similar_images(image_path)
            if similar_products:
                response_parts.append("Here are similar products I found:")
                for idx, product in enumerate(similar_products, 1):
                    similarity_percentage = round(product['similarity_score'] * 100, 2)
                    response_parts.append(
                        f"{idx}. {product['url']} (Similarity: {similarity_percentage}%)"
                    )
            else:
                response_parts.append("I couldn't find any similar products for the provided image.")

        if user_input:
            documents_retrieved = self.vectorDB.similarity_search(user_input, threshold=0.9)

            if not documents_retrieved:
                response_parts.append(
                    "I don't have any information on that particular product. Could you please specify another product or feature you're interested in?")
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

    def run(self):
        print("\nMultimodal Shopping Assistant is ready!")
        print("Available commands:")
        print("1. 'text' - Enter text query")
        print("2. 'voice' - Use voice input")
        print("3. 'image' - Search with an image")
        print("4. 'exit' - Quit the assistant")

        try:
            while True:
                print("\nSelect input method (text/voice/image/exit):", end=" ")
                input_method = input().lower().strip()

                if input_method == 'exit':
                    print("Exiting assistant.")
                    break

                user_input = None
                image_path = None

                if input_method == 'voice':
                    user_input = self.get_voice_input()
                elif input_method == 'text':
                    user_input = self.get_text_input()
                elif input_method == 'image':
                    image_path = input("Please enter the path to your image: ")
                    supplementary_text = input("Would you like to add any text description? (Press Enter to skip): ")
                    user_input = supplementary_text if supplementary_text.strip() else None
                else:
                    print("Invalid input method. Please try again.")
                    continue

                if user_input or image_path:
                    response = self.process_query(user_input, image_path)
                    print('-' * 80)
                    print('Assistant:', response)
                    print('-' * 80)

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nThank you for using Multimodal Shopping Assistant. Goodbye!")


def main():
    input_dict = {
        'vectorDB_path': 'faiss_index/',
        'image_index_path': 'image_faiss.index',
        'product_urls_path': 'products_url.pkl'
    }
    assistant = MultimodalShoppingAssistant(**input_dict)
    assistant.run()


if __name__ == "__main__":
    main()