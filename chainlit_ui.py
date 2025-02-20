import chainlit as cl
import speech_recognition as sr
from main import MultimodalShoppingAssistant
from backend.assistants.voice import VoiceHandler
from backend.assistants.image import ImageHandler
from backend.assistants.model import ChatbotHandler
from backend.config.config import Config

config = Config()

faiss_index = config.get('paths', 'faiss_index')
image_index = config.get('paths', 'image_index')
product_index = config.get('paths', 'product_index')

@cl.on_chat_start
async def start_chat():
    assistant = MultimodalShoppingAssistant(faiss_index, image_index, product_index)
    cl.user_session.set("assistant", assistant)
    await cl.Message(content="Welcome to the Multimodal Shopping Assistant! How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    assistant = cl.user_session.get("assistant")
    user_input = message.content
    image_path = None

    ## check if an image is uploaded ##
    if message.elements:
        for element in message.elements:
            if element.type == "image":
                image_path = element.path
                break
    response = assistant.process_query(user_input, image_path)
    await cl.Message(content=response).send()

@cl.on_audio_chunk
async def handle_audio(audio_chunk):
    assistant = cl.user_session.get("assistant")
    recognizer = sr.Recognizer()
    try:
        audio = sr.AudioData(audio_chunk)
        text = recognizer.recognize_google(audio)
        await cl.Message(content=f"You said: {text}").send()

        response = assistant.process_query(text)
        await cl.Message(content=response).send()
    except sr.UnknownValueError:
        await cl.Message(content="Could not understand audio").send()
    except sr.RequestError as e:
        await cl.Message(content=f"Error processing audio: {e}").send()

