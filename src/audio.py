import os
import subprocess
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def process_audio(video_path: str) -> list:
    """Extracts audio, transcribes it via Gemini, and formats it for ChromaDB."""
    print("\n PHASE 2: Audio Processing started...")
    audio_path = "data/input_videos/extracted_audio.mp3"
    
    try:
        # 1. Extract the audio track quietly using ffmpeg
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        print("   -> Ripping audio track...")
       
        command = [
            "ffmpeg", 
            "-i", video_path, 
            "-vn",              # Skip video
            "-acodec", "libmp3lame", 
            "-q:a", "2", 
            "-ac", "1",         # Force mono (easier for Gemini to process)
            "-ar", "44100",     # Standard sample rate
            audio_path, 
            "-y"
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(audio_path):
            print("   -> No audio track found in this video.")
            return []

        # 2. Upload the audio file directly to Gemini's File API
        print("   -> Uploading audio to Gemini...")
        audio_file = client.files.upload(file=audio_path)
        
        # 3. Prompt for strict timestamped transcription
        print("   -> Generating timestamped transcript...")
        prompt = (
            "Transcribe this audio file. Break it into short, logical sentences. "
            "For EVERY sentence, put the exact start time in MM:SS format at the beginning, like this:\n"
            "[00:05] This is the first sentence.\n"
            "[00:08] And here is the second sentence.\n"
            "Only output the timestamps and text. Do not add any extra formatting or conversational text."
        )
        
        # We can use the ultra-fast Lite model for audio!
        response = client.models.generate_content(
            # Change this in BOTH src/vlm.py and src/audio.py
            model='gemini-2.5-flash-lite', # CHANGED to the Pro model bucket
            contents=[prompt, audio_file]
        )
        # 4. Parse the output so ChromaDB can read it like a "frame"
        # 4. Parse the output to get pure seconds and text
        audio_metadata = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('[') and ']' in line:
                parts = line.split(']', 1)
                timestamp_part = parts[0].replace('[', '').strip()
                text_part = parts[1].strip()
                
                if text_part:
                    # Math trick to convert "MM:SS" into raw seconds
                    time_parts = timestamp_part.split(':')
                    seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                    
                    audio_metadata.append({
                        "timestamp_sec": seconds,
                        "text": text_part
                    })
                
        print(f" Extracted and transcribed {len(audio_metadata)} audio segments!")
        return audio_metadata

    except Exception as e:
        print(f" Audio processing error: {e}")
        return []