import streamlit as st
from inference import run_inference
import os

st.title("Lip Sync Inference App")

audio_file = st.file_uploader("Upload Audio File", type=['wav'])

if audio_file is not None:
    # Save uploaded audio to temp
    temp_audio_path = "temp/uploaded_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    if st.button("Run Inference"):
        with st.spinner("Processing... This may take a few minutes."):
            try:
                output_video = run_inference(temp_audio_path)
                if os.path.exists(output_video):
                    st.success("Inference completed!")
                    st.video(output_video)
                else:
                    st.error("Inference failed. Check logs.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
