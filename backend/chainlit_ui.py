import chainlit as cl
from chainlit.input_widget import Select, Switch
from chainlit.action import Action
import tempfile
import os
from typing import Optional
import speech_recognition as sr
from model import MultimodalShoppingAssistant
import sounddevice as sd
import soundfile as sf
import numpy as np

input_dict = {
    'vectorDB_path': 'faiss_index',
    'image_index_path': 'image_index/image_faiss.index',
    'product_urls_path': 'image_index/products_url.pkl'
}

SAMPLE_RATE = 44100
recording = None
input_mode = "text"


@cl.on_chat_start
async def start():
    assistant = MultimodalShoppingAssistant(**input_dict)

    cl.user_session.set("assistant", assistant)

    # Create input mode selector
    await cl.Action(
        name="input_mode",
        label="🔄 Input Mode",
        description="Select input method",
        actions=[
            "Text Input",
            "Voice Input",
            "Image Search"
        ]
    ).send()

    await cl.Message(
        content="Welcome to the Multimodal Shopping Assistant! Choose your input method above and start shopping!",
        actions=[
            cl.Action(name="start_recording", label="🎤 Start Recording", description="Click to start voice recording"),
            cl.Action(name="stop_recording", label="⏹️ Stop Recording", description="Click to stop voice recording")
        ]
    ).send()


@cl.action_callback("input_mode")
async def handle_input_mode(action):
    global input_mode
    input_mode = action.value.lower().replace(" ", "_")

    if input_mode == "voice_input":
        msg = "Voice input mode activated. Click the microphone button to start recording."
    elif input_mode == "image_search":
        msg = "Image search mode activated. Upload an image to search for similar products."
    else:
        msg = "Text input mode activated. Type your query in the chat box."

    await cl.Message(content=msg).send()


async def record_audio(duration=5):
    """Record audio using sounddevice"""
    recording = sd.rec(int(duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype=np.int16)
    sd.wait()
    return recording


@cl.action_callback("start_recording")
async def on_start_recording():
    if input_mode != "voice_input":
        await cl.Message(content="Please switch to Voice Input mode first!").send()
        return

    await cl.Message(content="🎤 Recording started... (5 seconds)").send()

    # Record audio
    recording = await record_audio()

    # Save recording to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        sf.write(temp_file.name, recording, SAMPLE_RATE)

        # Process the recording using the assistant
        assistant = cl.user_session.get("assistant")
        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_file.name) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                await cl.Message(content=f"Recognized text: {text}").send()

                # Process the query
                response = assistant.process_query(text)
                await cl.Message(content=response).send()

            except sr.UnknownValueError:
                await cl.Message(content="Sorry, I couldn't understand the audio.").send()
            except sr.RequestError as e:
                await cl.Message(content=f"Error with the speech recognition service: {e}").send()

        # Clean up temp file
        os.unlink(temp_file.name)


@cl.on_message
async def main(message: cl.Message):
    assistant = cl.user_session.get("assistant")

    if input_mode == "text_input":
        # Handle text input
        response = assistant.process_query(message.content)
        await cl.Message(content=response).send()

    elif input_mode == "image_search":
        # Handle image input
        if not message.elements:
            await cl.Message(content="Please upload an image to search.").send()
            return

        # Save uploaded image to temporary file
        image_element = message.elements[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            with open(temp_file.name, 'wb') as f:
                f.write(image_element.content)

            # Process image search with optional text query
            response = assistant.process_query(message.content, temp_file.name)
            await cl.Message(content=response).send()

            # Clean up temp file
            os.unlink(temp_file.name)


@cl.on_stop
def on_stop():
    # Cleanup when the application stops
    assistant = cl.user_session.get("assistant")
    if assistant:
        # Add any cleanup code here if needed
        pass

