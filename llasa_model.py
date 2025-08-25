from transformers import AutoProcessor, AutoModelForTextToWaveform
import torch
import soundfile as sf
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# --- Model Configuration ---
# Using a global cache to avoid reloading the model on every call in a stateless environment.
MODEL_CACHE = None

def get_model():
    """Loads and caches the Llasa-3B model and processor."""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        print("Loading Llasa-3B model and processor for the first time...")
        model_id = "HKUSTAudio/Llasa-3B"
        processor = AutoProcessor.from_pretrained(model_id, token=HUGGING_FACE_TOKEN)
        model = AutoModelForTextToWaveform.from_pretrained(model_id, token=HUGGING_FACE_TOKEN)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        MODEL_CACHE = {"processor": processor, "model": model, "device": device}
        print(f"Model loaded successfully on device: {device}")
    return MODEL_CACHE

def generate_audio(text: str, reference_audio_path: str) -> np.ndarray:
    """
    Generates audio from text using a reference audio file for voice cloning.

    Args:
        text (str): The input text to be converted to speech.
        reference_audio_path (str): The file path to the reference audio clip.

    Returns:
        np.ndarray: The generated audio waveform as a NumPy array.
    """
    model_components = get_model()
    processor = model_components["processor"]
    model = model_components["model"]
    device = model_components["device"]

    try:
        # Load the reference audio using soundfile
        reference_speech, sample_rate = sf.read(reference_audio_path)
        
        # Ensure the audio is in the correct format (mono)
        if reference_speech.ndim > 1:
            reference_speech = np.mean(reference_speech, axis=1)

        # Process the inputs for the model
        inputs = processor(
            text=text, 
            audios=reference_speech, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Generate the waveform
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        
        # Move the output to the CPU and convert to a NumPy array
        waveform = output.cpu().numpy().squeeze()
        
        return waveform, model.config.sampling_rate

    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        return np.array([]), 16000 # Return empty array and default sample rate on error
