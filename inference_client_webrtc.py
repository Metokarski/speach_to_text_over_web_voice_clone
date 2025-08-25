import streamlit as st
import requests
import websockets
import asyncio
import base64
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import json
import io
from ocr_component import ocr_component

# --- Configuration ---
SERVER_URL = "ws://localhost:8000"
HTTP_SERVER_URL = "http://localhost:8000"

# --- Main Application ---
st.set_page_config(layout="wide")
st.title("Llasa-3B Text-to-Speech WebApp")

st.markdown("""
This application uses the Llasa-3B model to generate speech from text, using a reference audio file for voice cloning.
""")

# --- State Management ---
if "ws" not in st.session_state:
    st.session_state.ws = None
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = asyncio.Queue()

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("1. Upload Reference Audio")
    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        if st.button("Upload and Set as Reference"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{HTTP_SERVER_URL}/upload_reference_audio", files=files)
                if response.status_code == 200:
                    st.success(f"Reference audio '{uploaded_file.name}' uploaded successfully!")
                else:
                    st.error(f"Error uploading file: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

    st.header("2. Input Text")
    text_input = st.text_area("Type text here", height=150)

    if st.button("Generate Speech"):
        if text_input:
            async def send_text():
                try:
                    if st.session_state.ws is None or st.session_state.ws.closed:
                        st.session_state.ws = await websockets.connect(f"{SERVER_URL}/audio")
                    
                    await st.session_state.ws.send(json.dumps({"text": text_input}))
                except Exception as e:
                    st.error(f"Failed to send text: {e}")

            asyncio.run(send_text())
        else:
            st.warning("Please enter some text to generate speech.")

    st.header("3. Camera Text Input (OCR)")
    use_camera = st.toggle("Enable Camera OCR")
    if use_camera:
        ocr_text = ocr_component()
        if ocr_text and st.button("Generate from OCR Text"):
            async def send_ocr_text():
                try:
                    if st.session_state.ws is None or st.session_state.ws.closed:
                        st.session_state.ws = await websockets.connect(f"{SERVER_URL}/audio")
                    
                    await st.session_state.ws.send(json.dumps({"text": ocr_text}))
                except Exception as e:
                    st.error(f"Failed to send OCR text: {e}")
            asyncio.run(send_ocr_text())


# --- Audio Playback ---
st.header("Generated Audio")
audio_placeholder = st.empty()

async def audio_listener():
    """Listens for incoming audio from the WebSocket and plays it."""
    while True:
        try:
            if st.session_state.ws is None or st.session_state.ws.closed:
                await asyncio.sleep(1)
                continue

            message = await st.session_state.ws.recv()
            
            if message.startswith("data:audio/raw;base64,"):
                audio_b64 = message.split(",")[1]
                audio_bytes = base64.b64decode(audio_b64)
                
                # Assuming the server sends 16kHz mono audio
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Use a BytesIO buffer to work with st.audio
                buffer = io.BytesIO()
                sf.write(buffer, audio_array, 16000, format='WAV')
                buffer.seek(0)
                
                audio_placeholder.audio(buffer, format='audio/wav')

        except websockets.exceptions.ConnectionClosed:
            st.warning("WebSocket connection closed. Reconnecting...")
            st.session_state.ws = None
            await asyncio.sleep(2)
        except Exception as e:
            st.error(f"An error occurred in the audio listener: {e}")
            await asyncio.sleep(2)

# --- Main Execution ---
# This part is tricky in Streamlit's execution model.
# A common pattern is to run the async listener in a separate thread.
import threading

def run_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def main():
    # Start the WebSocket connection when the app loads
    if st.session_state.ws is None:
        try:
            st.session_state.ws = await websockets.connect(f"{SERVER_URL}/audio")
        except Exception as e:
            st.error(f"Could not connect to WebSocket server: {e}")
    
    # Run the listener
    await audio_listener()


if "async_thread_started" not in st.session_state:
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=run_async_loop, args=(loop,), daemon=True)
    t.start()
    
    # Schedule the main async function to run in the new loop
    asyncio.run_coroutine_threadsafe(main(), loop)
    
    st.session_state.async_thread_started = True

st.info("Ready to generate audio. Use the controls in the sidebar.")
