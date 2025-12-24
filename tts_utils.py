from pydub import AudioSegment
from chatterbox.tts_turbo import ChatterboxTurboTTS
import torch
import torchaudio
import numpy as np
import time
from functools import lru_cache
import soundfile as sf
import os
from huggingface_hub import login
import sys

@lru_cache(maxsize=None)  # Cache all results (or set a maxsize limit)
def load_pipeline(lang_code='en'):
    try:
        # Authenticate with Hugging Face if token is provided in environment
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ChatterboxTurboTTS on {device}...")
        return ChatterboxTurboTTS.from_pretrained(device=device)
    except Exception as e:
        print(f"Failed to load ChatterboxTurboTTS: {e}", file=sys.stderr)
        raise

def generate_audio(text, pipeline, voice_option="default.wav", sample_rate=24000):
    # Determine voice path. Chatterbox Turbo requires a reference clip.
    voice_path = voice_option
    
    # Logic to find the voice file
    found = False
    
    # 1. Check direct path
    if os.path.exists(voice_path) and os.path.isfile(voice_path):
        found = True
    
    # 2. Check in 'voices' folder
    if not found:
        candidate = os.path.join("voices", voice_option)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            voice_path = candidate
            found = True
            
    # 3. Check in 'voices' folder with .wav extension appended
    if not found:
        candidate_ext = os.path.join("voices", f"{voice_option}.wav")
        if os.path.exists(candidate_ext) and os.path.isfile(candidate_ext):
            voice_path = candidate_ext
            found = True

    if not found:
        # Better to raise error so user knows they need a file.
        error_msg = f"Voice reference file '{voice_option}' not found. Please upload a .wav file to the 'voices' folder."
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)

    print(f"Generating audio using voice prompt: {voice_path}")
    
    try:
        time_start = time.time()
        
        # Generate audio (returns tensor)
        wav = pipeline.generate(text, audio_prompt_path=voice_path)
        
        time_end = time.time()
        
        if wav is None:
            raise RuntimeError("Model returned None. Generation failed.")
            
        # Convert torch tensor to numpy for soundfile
        audio_data = wav.cpu().numpy()
        
        # Squeeze dimensions if needed (1, N) -> (N,)
        if audio_data.ndim == 2:
            audio_data = audio_data.squeeze()
            
        print("Audio generation completed successfully")
        
        # Display results
        return audio_data, time_end - time_start
                    
    except Exception as e:
        print(f"Error occurred in generating audio: {e}", file=sys.stderr)
        # Re-raise so app.py catches it and displays the message to the user
        raise RuntimeError(f"Generation Error: {str(e)}")

def convert_wav_to_mp3(input_filepath, output_filepath="audio.mp3"):
    """
    Converts a WAV audio file to MP3 format with good compression for smaller file size
    while trying to maintain reasonable audio quality.

    Args:
        filepath (str): The path to the input WAV audio file.
        output_filepath (str, optional): The desired path for the output MP3 file.
                                         Defaults to "audio.mp3" in the current directory.
    """
    try:
        audio = AudioSegment.from_wav(input_filepath)
        audio.export(output_filepath, format="mp3", bitrate="64k") # 128k is a good balance
        print(f"Successfully converted '{input_filepath}' to '{output_filepath}'")
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)

def save_audio(audio_data, output_path_folder="static/", sampling_rate=24000, page_numbers=[1,20], voice_option="af_heart", book_name="book"):
    # Sanitize voice option name for filename
    voice_name = os.path.basename(voice_option)
    voice_name = os.path.splitext(voice_name)[0]
    
    high_quality_audio_path = os.path.join(output_path_folder, f"{book_name} part x {voice_name} pg{page_numbers[0]}-{page_numbers[1]}.wav")
    low_quality_audio_path = os.path.join(output_path_folder, f"{book_name} part x {voice_name} pg{page_numbers[0]}-{page_numbers[1]}.mp3")
    
    if os.path.exists(high_quality_audio_path):
        os.remove(high_quality_audio_path)
    if os.path.exists(low_quality_audio_path):
        os.remove(low_quality_audio_path)
        
    try:
        sf.write(high_quality_audio_path, audio_data, sampling_rate)
        convert_wav_to_mp3(output_filepath=low_quality_audio_path, input_filepath=high_quality_audio_path)
        print("Audio file saved successfully")
        return high_quality_audio_path, low_quality_audio_path
    except Exception as e:
        print(f"Error saving audio file: {e}", file=sys.stderr)
        raise