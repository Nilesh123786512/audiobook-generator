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
import re # Added regex module

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

def split_text(text, max_length=500):
    """
    Splits text into chunks respecting sentence boundaries to avoid model context overflow.
    """
    chunks = []
    current_chunk = ""
    # Split by sentence endings (.?!) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        # Hard limit split: if a single sentence is huge (unlikely but possible), force split it
        if len(sentence) > max_length:
             # If current chunk has content, push it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Add the long sentence as its own chunk (or split further if needed, but keeping simple here)
            chunks.append(sentence.strip())
            continue

        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

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
        # Fallback to default generation if file not found
        print(f"Voice reference file '{voice_option}' not found. Using default voice generation.", file=sys.stderr)
        voice_path = None

    if voice_path:
        print(f"Generating audio using voice prompt: {voice_path}")
    else:
        print("Generating audio using default voice")
    
    try:
        time_start = time.time()
        
        # Step 1: Split text into chunks
        chunks = split_text(text)
        print(f"Processing: Text split into {len(chunks)} chunks for generation.")
        
        all_audio_segments = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # print(f"Generating chunk {i+1}/{len(chunks)}...") # Optional loop log
            
            # Generate audio for chunk
            if voice_path:
                wav = pipeline.generate(chunk, audio_prompt_path=voice_path)
            else:
                wav = pipeline.generate(chunk)
            
            if wav is None:
                print(f"Warning: Chunk {i+1} failed to generate.")
                continue

            chunk_data = wav.cpu().numpy()
            
            # Squeeze dimensions if needed
            if chunk_data.ndim == 2:
                chunk_data = chunk_data.squeeze()
                
            all_audio_segments.append(chunk_data)
            
            # Add a short pause between sentences/chunks (e.g., 0.25 seconds)
            silence_length = int(0.25 * sample_rate)
            all_audio_segments.append(np.zeros(silence_length))

        if not all_audio_segments:
             raise RuntimeError("Model returned None for all chunks. Generation failed.")

        # Step 2: Concatenate all audio segments
        audio_data = np.concatenate(all_audio_segments)
        
        time_end = time.time()
        
        # DEBUG: Check if audio is silence
        print(f"DEBUG: Audio Shape: {audio_data.shape}")
        print(f"DEBUG: Audio Amplitude - Max: {audio_data.max():.4f}, Min: {audio_data.min():.4f}")
        if np.abs(audio_data).max() == 0:
            print("DEBUG: WARNING - Generated audio is pure silence!")

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
        # Ensure we don't crash if ffmpeg is missing, but maybe warn user
        print("Tip: Install FFmpeg and add it to PATH for MP3 conversion to work.")

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
        # Only try converting if wav write succeeded
        if os.path.exists(high_quality_audio_path):
            convert_wav_to_mp3(output_filepath=low_quality_audio_path, input_filepath=high_quality_audio_path)
            
        print("Audio file saved successfully")
        return high_quality_audio_path, low_quality_audio_path
    except Exception as e:
        print(f"Error saving audio file: {e}", file=sys.stderr)
        raise