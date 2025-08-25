import os
import base64
import uvicorn
import traceback
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf

from llasa_model import generate_audio

# --- Helper Functions ---
def print_colored(text, color):
    """Prints text in a specified color."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "grey": "\033[90m",
        "end": "\033[0m",
    }
    color_code = colors.get(color, "")
    print(f"{color_code}{text}{colors['end']}")

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
        print_colored(f"Reference audio updated to: {current_reference_audio}", "green")
        return {"message": f"File '{file.filename}' uploaded successfully.", "status": "success"}
    except Exception as e:
        print_colored(f"Error uploading file: {str(e)}", "red")
        return {"message": f"Error uploading file: {str(e)}", "status": "error"}

# --- WebSocket Endpoint ---
@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles the real-time text-to-speech WebSocket connection.
    Receives text, generates audio, and streams it back to the client.
    """
    await websocket.accept()
    print_colored("WebSocket connection established.", "cyan")
    
    try:
        while True:
            # Receive JSON data from the client
            data = await websocket.receive_json()
            text = data.get("text")

            if not text:
                continue

            if current_reference_audio is None:
                print_colored("Warning: No reference audio has been uploaded.", "yellow")
                # Optionally, send a warning back to the client
                await websocket.send_json({"error": "Please upload a reference audio file first."})
                continue

            print_colored(f"Received text for TTS: '{text}'", "grey")

            # Generate audio using the Llasa-3B model
            waveform, sample_rate = generate_audio(text, current_reference_audio)

            if waveform.size == 0:
                print_colored("Audio generation failed. Skipping.", "red")
                continue

            # Convert audio to 16-bit PCM format
            waveform_int16 = (waveform * 32767).astype(np.int16)

            # Encode the audio data in Base64
            processed_data = base64.b64encode(waveform_int16.tobytes()).decode('utf-8')

            # Send the processed audio back to the client
            await websocket.send_text(f"data:audio/raw;base64,{processed_data}")
            print_colored(f"Sent {len(waveform_int16) / sample_rate:.2f}s of audio to client.", "grey")

    except Exception as e:
        print_colored(f"WebSocket error: {e}", "red")
        print_colored(f"Full traceback:\n{traceback.format_exc()}", "red")
    finally:
        print_colored("WebSocket connection closed.", "cyan")
        await websocket.close()

# --- Main Execution ---
if __name__ == "__main__":
    print_colored("Starting FastAPI server...", "green")
    # It's recommended to run the server with `uvicorn inference_server:app --reload`
    # from the command line for development.
    uvicorn.run(app, host="0.0.0.0", port=8000)
