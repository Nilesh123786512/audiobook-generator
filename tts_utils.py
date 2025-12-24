from pydub import AudioSegment
from chatterbox.tts_turbo import ChatterboxTurboTTS
import torch
import torchaudio
import numpy as np
import time
from functools import lru_cache
import soundfile as sf
import os

@lru_cache(maxsize=None)  # Cache all results (or set a maxsize limit)
def load_pipeline(lang_code='en'):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return ChatterboxTurboTTS.from_pretrained(device=device)
    except Exception as e:
        print(f"Failed to load ChatterboxTurboTTS: {e}")
        # Re-raise or return None/custom error object
        raise # or return None

def generate_audio(text,pipeline,voice_option="default.wav",sample_rate=24000):
    try:
            # Determine voice path. Chatterbox Turbo requires a reference clip.
            voice_path = voice_option
            if not os.path.exists(voice_path):
                 # Check in 'voices' folder
                 candidate = os.path.join("voices", voice_option)
                 if os.path.exists(candidate):
                     voice_path = candidate
                 elif os.path.exists(candidate + ".wav"):
                     voice_path = candidate + ".wav"
                 else:
                     print(f"Voice file not found for option: {voice_option}")
                     return None, 0

            print(f"Generating audio using voice prompt: {voice_path}")
            time_start=time.time()
            
            # Generate audio (returns tensor)
            wav = pipeline.generate(text, audio_prompt_path=voice_path)
            
            time_end=time.time()
            
            if wav is None:
                print("No audio generated")
                return None, 0
                
            # Convert torch tensor to numpy for soundfile
            audio_data = wav.cpu().numpy()
            if audio_data.ndim == 2:
                audio_data = audio_data.squeeze()
                
            print("Audio generation completed successfully")
            
            # Display results
            return audio_data, time_end-time_start
                    
                    
    except Exception as e:
        # st.error(f"Error generating audio: {str(e)}")
        print(f"Error occured in generating audio:{e}")
        return None, 0


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
        print(f"Error during conversion: {e}")

def save_audio(audio_data,output_path_folder="static/",sampling_rate=24000,page_numbers=[1,20],voice_option="af_heart",book_name="book"):
    # Sanitize voice option name for filename
    voice_name = os.path.basename(voice_option)
    voice_name = os.path.splitext(voice_name)[0]
    
    high_quality_audio_path=os.path.join(output_path_folder, f"{book_name} part x {voice_name} pg{page_numbers[0]}-{page_numbers[1]}.wav")#output_path_folder+"audio.wav"
    low_quality_audio_path=os.path.join(output_path_folder, f"{book_name} part x {voice_name} pg{page_numbers[0]}-{page_numbers[1]}.mp3")#output_path_folder+"audio.mp3"
    if os.path.exists(high_quality_audio_path):
        os.remove(high_quality_audio_path)
    if os.path.exists(low_quality_audio_path):
        os.remove(low_quality_audio_path)
    sf.write(high_quality_audio_path,audio_data,sampling_rate)
    convert_wav_to_mp3(output_filepath=low_quality_audio_path,input_filepath=high_quality_audio_path)
    print("Audio file saved successfully")
    return high_quality_audio_path,low_quality_audio_path