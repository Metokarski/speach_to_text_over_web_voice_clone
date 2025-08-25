import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
from PIL import Image
import time

# In a real implementation, you would use a proper OCR library.
# For this placeholder, we'll simulate OCR by allowing the user to type what they see.
# This demonstrates the data flow.
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

def ocr_from_frame(frame: np.ndarray) -> str:
    """
    Performs OCR on a single video frame.
    
    Args:
        frame (np.ndarray): The video frame.
        
    Returns:
        str: The recognized text.
    """
    if not PYTESSERACT_AVAILABLE:
        return "pytesseract not installed. Please install it to use OCR."
        
    try:
        # pytesseract works with PIL Images
        image = Image.fromarray(frame)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"OCR Error: {e}"

def ocr_component():
    """
    A Streamlit component to capture video and perform OCR.
    """
    st.markdown("### Camera Text Recognition")
    
    webrtc_ctx = webrtc_streamer(
        key="ocr-streamer",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=None, # We will process frames manually
        async_processing=True,
    )

    text_placeholder = st.empty()
    ocr_button = st.button("Recognize Text from Camera Feed")

    if ocr_button and webrtc_ctx.video_receiver:
        try:
            frame = webrtc_ctx.video_receiver.get_frame(timeout=10)
            img_rgb = frame.to_ndarray(format="rgb24")
            
            with st.spinner("Performing OCR..."):
                recognized_text = ocr_from_frame(img_rgb)
            
            text_placeholder.text_area("Recognized Text", value=recognized_text, height=200)
            st.image(img_rgb, caption="Last captured frame")

        except Exception as e:
            st.error(f"Could not get frame from camera: {e}")
    elif ocr_button:
        st.warning("Camera not active. Please start the stream first.")

    if not PYTESSERACT_AVAILABLE:
        st.warning("`pytesseract` is not installed. OCR functionality is disabled. Please run `pip install pytesseract` and install the Tesseract engine.")

    # This component doesn't directly return the text in this structure,
    # but a more advanced implementation with session state could.
    # For now, the user can copy-paste from the text area.
