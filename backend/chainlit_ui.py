import chainlit as cl
from model import MultimodalShoppingAssistant  # Replace with your actual module name
import tempfile
import os

input_dict = {
    'vectorDB_path': 'faiss_index',
    'image_index_path': 'image_index/image_faiss.index',
    'product_urls_path': 'image_index/products_url.pkl'
}
assistant = MultimodalShoppingAssistant(**input_dict)

@cl.on_chat_start
async def start_chat():
    welcome_message = """Welcome to the Multimodal Shopping Assistant! 🛍️

You can interact with me in the following ways:
- **Text Input**: Type your query in the chat box.
- **Voice Input**: Click the 🎤 button to speak.
- **Image Input**: Upload an image to find similar products.

How can I assist you today?"""

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content
    image_path = None

    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(element.content)
                    image_path = temp_file.name

    response = assistant.process_query(user_input, image_path)

    await cl.Message(content=response).send()

    if image_path and os.path.exists(image_path):
        os.remove(image_path)


@cl.action_callback("voice_input")
async def on_voice_input(action: cl.Action):
    voice_input = assistant.get_voice_input()
    if voice_input:
        await cl.Message(content=f"🎤 You said: {voice_input}").send()
        response = assistant.process_query(voice_input)
        await cl.Message(content=response).send()
    else:
        await cl.Message(content="Sorry, I couldn't understand your voice input. Please try again.").send()


@cl.action_callback("image_input")
async def on_image_input(action: cl.Action):
    await cl.Message(content="Please upload an image to find similar products.").send()


@cl.action_callback("exit")
async def on_exit(action: cl.Action):
    await cl.Message(content="Thank you for using the Multimodal Shopping Assistant. Goodbye!").send()
    await cl.close()


@cl.on_chat_end
async def on_chat_end():
    await cl.Message(content="Session ended. Thank you for using the Multimodal Shopping Assistant!").send()


async def main():
    actions = [
        cl.Action(name="voice_input", value="voice_input", label="🎤 Voice Input"),
        cl.Action(name="image_input", value="image_input", label="🖼️ Image Input"),
        cl.Action(name="exit", value="exit", label="🚪 Exit")
    ]

    await cl.Message(content="Choose an input method:", actions=actions).send()

if __name__ == "__main__":
    cl.run(main)