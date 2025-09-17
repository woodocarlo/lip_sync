import os
import time
import librosa
import numpy as np
import cv2
import shutil
import soundfile as sf
import subprocess
from lipsync import LipSync

# Setup directories
os.makedirs('output/final_inference', exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Input files
input_video = 'input/person.mp4'
input_audio = 'input/test3.wav' 
weights_path = 'weights/wav2lip.pth'

# ===== SINGLE FINAL CONFIGURATION =====
final_config = {
    'id': 1,
    'name': 'FINAL_INFERENCE',
    'params': {'nosmooth': True, 'img_size': 96, 'pad_top': 0, 'pad_bottom': 15, 'pad_left': 0, 'pad_right': 0},
    'processing': {'post_process_blending': True, 'keysync_approach': True},
    'description': 'Single configuration for final inference (no silent audio processing)'
}

# ===== POST PROCESSING =====
def improved_post_process_blending(video_path, output_path, blend_alpha=0.1):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frames_buffer = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame = frame.astype(np.uint8)
            frames_buffer.append(frame)
        cap.release()
        for i, current_frame in enumerate(frames_buffer):
            if i > 0:
                prev_frame = frames_buffer[i-1]
                if current_frame.shape == prev_frame.shape and current_frame.dtype == prev_frame.dtype:
                    blended_frame = cv2.addWeighted(current_frame, 1-blend_alpha, prev_frame, blend_alpha, 0)
                else:
                    blended_frame = current_frame
                out.write(blended_frame)
            else:
                out.write(current_frame)
        out.release()
        print(f"  Post-processing completed: {len(frames_buffer)} frames processed")
        return True
    except Exception as e:
        print(f"Post-processing error: {e}")
        shutil.copyfile(video_path, output_path)
        return False

def keysync_approach_processing(video_path, audio_path, output_path, lip_params):
    try:
        lip = LipSync(
            model='wav2lip',
            checkpoint_path=weights_path,
            nosmooth=lip_params['nosmooth'],
            img_size=lip_params['img_size'],
            pad_top=lip_params['pad_top'],
            pad_bottom=lip_params['pad_bottom'],
            pad_left=lip_params['pad_left'],
            pad_right=lip_params['pad_right'],
            device='cpu'
        )
        lip.sync(video_path, audio_path, output_path)
        return True
    except Exception as e:
        print(f"Error in KeySync processing: {e}")
        return False

# ===== AUDIO + VIDEO MERGE =====
def merge_audio_video(video_path, audio_path, output_path):
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            output_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error merging audio and video: {e}")
        shutil.copyfile(video_path, output_path)
        return False

# ===== FINAL INFERENCE FUNCTION =====
def run_final_inference():
    config = final_config
    params = config['params']
    processing = config['processing']

    print(f"\n--- Final Inference: {config['name']} ---")
    print(f"Description: {config['description']}")
    print(f"Parameters: {params}")

    start_time = time.time()
    raw_output = "temp/final_temp.mp4"
    final_output = "output/final_inference/final_output.mp4"

    try:
        # step 1: run lipsync
        if processing['keysync_approach']:
            success = keysync_approach_processing(input_video, input_audio, raw_output, params)
            if not success:
                raise Exception("KeySync processing failed")
            audio_to_use = input_audio
        else:
            audio_to_use = input_audio
            lip = LipSync(
                model='wav2lip',
                checkpoint_path=weights_path,
                nosmooth=params['nosmooth'],
                img_size=params['img_size'],
                pad_top=params['pad_top'],
                pad_bottom=params['pad_bottom'],
                pad_left=params['pad_left'],
                pad_right=params['pad_right'],
                device='cpu'
            )
            lip.sync(input_video, audio_to_use, raw_output)

        # step 2: apply blending + merge audio
        if processing['post_process_blending']:
            temp_video = "temp/processed_video.mp4"
            improved_post_process_blending(raw_output, temp_video)
            merge_audio_video(temp_video, audio_to_use, final_output)
            if os.path.exists(temp_video):
                os.remove(temp_video)
        else:
            merge_audio_video(raw_output, audio_to_use, final_output)

        if os.path.exists(raw_output):
            os.remove(raw_output)

        processing_time = time.time() - start_time
        print(f"✓ SUCCESS! Final output saved: {final_output}")
        print(f"Processing time: {processing_time:.2f}s")

    except Exception as e:
        print(f"✗ FAILED: {str(e)}")

def run_inference(audio_path):
    global input_audio
    input_audio = audio_path
    run_final_inference()
    return "output/final_inference/final_output.mp4"

# ===== MAIN =====
if __name__ == "__main__":
    print("🚀 Starting final inference...")

    if not os.path.exists(input_video):
        print(f"❌ ERROR: Input video not found: {input_video}")
        exit(1)
    if not os.path.exists(input_audio):
        print(f"❌ ERROR: Input audio not found: {input_audio}")
        exit(1)
    if not os.path.exists(weights_path):
        print(f"❌ ERROR: Model weights not found: {weights_path}")
        exit(1)

    run_final_inference()
    print("\n🎉 FINAL INFERENCE COMPLETED! Check output/final_inference/")
