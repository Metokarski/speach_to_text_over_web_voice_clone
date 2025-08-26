import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
from PIL import Image
import time
import logging

# --- Professional Logging Setup ---
# (We can use a basic config here as Streamlit manages the main process)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [CLIENT - OCR] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    A Streamlit component to capture a single video frame and perform OCR.
    This "capture-on-demand" model is more efficient and avoids queue overflows.
    """
    st.markdown("### Camera Text Recognition")

    webrtc_ctx = webrtc_streamer(
        key="ocr-streamer",
        mode=WebRtcMode.SENDRECV,  # Allows us to send a frame back for display
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.write("When you see the text clearly in the video feed, click the button below.")
    ocr_button = st.button("Recognize Text from Camera Feed")

    if "recognized_text" not in st.session_state:
        st.session_state.recognized_text = ""
    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None

    if ocr_button and webrtc_ctx.video_receiver:
        logger.info("'Recognize Text' button clicked. Attempting to capture frame.")
        try:
            with st.spinner("Capturing frame and performing OCR..."):
                frame = webrtc_ctx.video_receiver.get_frame(timeout=10)
                img_rgb = frame.to_ndarray(format="rgb24")
                logger.info(f"Successfully captured a {img_rgb.shape[1]}x{img_rgb.shape[0]} frame.")

                # --- Performance Timing ---
                start_time = time.time()
                recognized_text = ocr_from_frame(img_rgb)
                end_time = time.time()
                processing_time = end_time - start_time
                logger.info(f"OCR processing took {processing_time:.2f} seconds.")
                logger.info(f"Recognized text: '{recognized_text[:100]}...'")

                # Store the captured image and recognized text in session state
                st.session_state.captured_image = img_rgb
                st.session_state.recognized_text = recognized_text

        except Exception as e:
            logger.error(f"Could not get frame from camera: {e}", exc_info=True)
            st.error(f"Could not get frame from camera: {e}")
    elif ocr_button:
        st.warning("Camera not active. Please start the stream first.")

    # Display the results from session state
    if st.session_state.captured_image is not None:
        st.image(st.session_state.captured_image, caption="Last Captured Frame")
    
    st.text_area(
        "Recognized Text", 
        value=st.session_state.recognized_text, 
        height=200,
        key="ocr_text_area"
    )

    if not PYTESSERACT_AVAILABLE:
        st.warning("`pytesseract` is not installed. OCR functionality is disabled. Please run `pip install pytesseract` and install the Tesseract engine.")

    # Return the recognized text so the main app can use it
    return st.session_state.recognized_text
