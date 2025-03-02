import time
from backend.assistants.voice import VoiceHandler
from backend.assistants.image import ImageHandler
from backend.assistants.model import ChatbotHandler
from backend.config.config import Config

config = Config()

faiss_index = config.get('paths', 'faiss_index')
image_index = config.get('paths', 'image_index')
product_index = config.get('paths', 'product_index')

class MultimodalShoppingAssistant:
    def __init__(self, vectorDB_path, image_index_path, product_urls_path):
        self.voice_handler = VoiceHandler()
        self.image_handler = ImageHandler(image_index_path, product_urls_path)
        self.chatbot_handler = ChatbotHandler(vectorDB_path)

    def process_query(self, user_input=None, image_path=None):
        response_parts = []

        if image_path:
            similar_products = self.image_handler.find_similar_images(image_path)
            if similar_products:
                response_parts.append("Here are similar products I found:")
                for idx, product in enumerate(similar_products, 1):
                    response_parts.append(
                        f"{idx}. {product['url']} (Similarity: {round(product['similarity_score'] * 100, 2)}%)")
            else:
                response_parts.append("No similar products found for the given image.")

        if user_input:
            response_parts.append(self.chatbot_handler.process_text_query(user_input))

        return "\n\n".join(response_parts) if response_parts else "No valid input received."

    def run(self):
        print("\nMultimodal Shopping Assistant is ready!")
        while True:
            input_method = input("Select input method (text/voice/image/exit): ").lower().strip()
            if input_method == 'exit':
                print("Exiting assistant.")
                break

            user_input, image_path = None, None
            if input_method == 'voice':
                user_input = self.voice_handler.get_voice_input()
            elif input_method == 'text':
                user_input = input("Enter your query: ").strip()
            elif input_method == 'image':
                image_path = input("Enter image path: ")
                user_input = input("Additional description (optional): ").strip() or None

            response = self.process_query(user_input, image_path)
            print('-' * 80)
            print('Assistant:', response)
            print('-' * 80)
            time.sleep(0.5)


def main():
    assistant = MultimodalShoppingAssistant(faiss_index,
                                            image_index,
                                            product_index)
    assistant.run()


if __name__ == "__main__":
    main()

