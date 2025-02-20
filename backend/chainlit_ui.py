# chainlit_app.py
import chainlit as cl
import sounddevice as sd
import numpy as np
import wave
from model import MultimodalShoppingAssistant


# Initialize the assistant when the chat starts
@cl.on_chat_start
async def start_chat():
    input_dict = {
        'vectorDB_path': 'faiss_index',
        'image_index_path': 'image_index/image_faiss.index',
        'product_urls_path': 'image_index/products_url.pkl'
    }
    assistant = MultimodalShoppingAssistant(**input_dict)
    cl.user_session.set("assistant", assistant)

    # Add a microphone icon button
    actions = [
        cl.Action(name="start_recording", value="start_recording", label="🎤", payload={"action": "start_recording"})
    ]
    await cl.Message(content="Click the microphone icon to start recording your voice.", actions=actions).send()


# Handle user messages
@cl.on_message
async def handle_message(message: cl.Message):
    assistant = cl.user_session.get("assistant")
    user_input = message.content
    image_path = None

    # Check if the user uploaded an image
    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                image_path = element.path
                break

    # Process the query
    response = assistant.process_query(user_input, image_path)
    await cl.Message(content=response).send()


# Handle button clicks using @cl.action_callback
@cl.action_callback("start_recording")
async def on_action_callback(action: cl.Action):
    await cl.Message(content="Recording started. Speak now...").send()

    # Audio recording settings
    sample_rate = 16000  # 16 kHz
    duration = 10  # Maximum recording duration in seconds

    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    audio_path = "user_audio.wav"
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16))

    # Process the audio file
    assistant = cl.user_session.get("assistant")
    user_input = assistant.get_voice_input(audio_path)

    if not user_input:
        await cl.Message(content="Sorry, I couldn't process the voice input. Please try again.").send()
        return

    # Process the query
    response = assistant.process_query(user_input)
    await cl.Message(content=response).send()

