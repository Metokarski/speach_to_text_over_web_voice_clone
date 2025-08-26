# speach_to_text_over_web_voice_clone

## Llasa-3B Text-to-Speech WebApp

This repository has been refactored to provide a web-based interface for the [HKUSTAudio/Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B) model, enabling text-to-speech with voice cloning from a reference audio file.

## Setup

1.  **Create and Activate a Virtual Environment**:
    It is highly recommended to use a virtual environment to manage project-specific dependencies.

    ```bash
    # Create the virtual environment
    python3 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate
    ```
    *On Windows, the activation command is `.\.venv\Scripts\activate`*

2.  **Install Dependencies**:
    Once the virtual environment is activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements_webrtc.txt
    ```
    For the camera-based OCR feature, you will also need to install Tesseract on your system.

3.  **Configure Environment Variables**:
    Create a `.env` file in the root of the project by copying the `.env.example` file:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and add your Hugging Face access token. This is required to download the Llasa-3B model.
    ```
    HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

## Usage

The application runs as two separate processes: a backend server and a frontend client. You must run both simultaneously in two separate terminals.

1.  **Run the Backend Server**:
    The server handles the model inference and file uploads.
    ```bash
    # For local development
    python -m uvicorn inference_server:app --reload

    # If running on a remote server, make it accessible externally
    python -m uvicorn inference_server:app --reload --host 0.0.0.0
    ```

2.  **Run the Frontend Client**:
    The client is a Streamlit web application.
    ```bash
    streamlit run inference_client_webrtc.py
    ```
    You can then access the application in your browser at the URL provided by Streamlit.

## Configuration

### Connecting to a Remote Server

If your backend server is running on a different machine (e.g., a cloud GPU instance), you need to tell the client how to connect to it. You can do this in one of two ways:

1.  **Using the `.env` file (Recommended)**:
    Set the `SERVER_IP` variable in your `.env` file to the public IP address of your server.
    ```
    SERVER_IP="000.000.000.00"
    ```

2.  **Using a Command-Line Argument**:
    You can specify the server's IP address directly when launching the client. This will override any value set in the `.env` file.
    ```bash
    streamlit run inference_client_webrtc.py -- --server_ip 000.000.000.00
    ```
    *(Note the extra `--` which is required to pass arguments to the Streamlit script).*
