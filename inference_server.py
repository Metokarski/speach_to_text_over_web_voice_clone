import os
import base64
import uvicorn
import traceback
import numpy as np
import logging
import time
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf

from llasa_model import generate_audio

# --- Professional Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
# For a production app, this state should be managed more robustly (e.g., per-session).
# For this demo, a simple global variable suffices.
current_reference_audio = None

# Ensure the prompts directory exists
os.makedirs("prompts", exist_ok=True)

# --- HTTP Endpoints ---
@app.post("/upload_reference_audio")
async def upload_reference_audio(file: UploadFile = File(...)):
    """
    Handles uploading of the reference audio file.
    The file is saved in the 'prompts' directory.
    """
    global current_reference_audio
    try:
        file_path = os.path.join("prompts", file.filename)
        
        # Read the content of the uploaded file
        content = await file.read()
        
        # Write the content to the new file
        with open(file_path, "wb") as f:
            f.write(content)
            
        current_reference_audio = file_path
        logger.info(f"Reference audio updated to: {current_reference_audio}")
        return {"message": f"File '{file.filename}' uploaded successfully.", "status": "success"}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        return {"message": f"Error uploading file: {str(e)}", "status": "error"}

# --- WebSocket Endpoint ---
@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles the real-time text-to-speech WebSocket connection.
    Receives text, generates audio, and streams it back to the client.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established with client: {websocket.client.host}:{websocket.client.port}")
    
    try:
        while True:
            # Receive JSON data from the client
            data = await websocket.receive_json()
            text = data.get("text")

            if not text:
                continue

            if current_reference_audio is None:
                logger.warning("TTS request received but no reference audio is set.")
                await websocket.send_json({"error": "Please upload a reference audio file first."})
                continue

            logger.info(f"Received text for TTS: '{text[:100]}...'")
            logger.debug(f"Using reference audio: {current_reference_audio}")

            # --- Performance Timing ---
            start_time = time.time()
            
            # Generate audio using the Llasa-3B model
            waveform, sample_rate = generate_audio(text, current_reference_audio)
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Audio generation took {processing_time:.2f} seconds.")

            if waveform.size == 0:
                logger.error("Audio generation failed, returned empty waveform.")
                continue

            # Convert audio to 16-bit PCM format
            waveform_int16 = (waveform * 32767).astype(np.int16)

            # Encode the audio data in Base64
            processed_data = base64.b64encode(waveform_int16.tobytes()).decode('utf-8')

            # Send the processed audio back to the client
            await websocket.send_text(f"data:audio/raw;base64,{processed_data}")
            
            duration_s = len(waveform_int16) / sample_rate
            logger.info(f"Sent {duration_s:.2f}s of audio to client.")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket connection closed with client: {websocket.client.host}:{websocket.client.port}")
        await websocket.close()

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    # It's recommended to run the server with `uvicorn inference_server:app --reload`
    # from the command line for development.
    uvicorn.run(app, host="0.0.0.0", port=8000)
