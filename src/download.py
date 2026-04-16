import yt_dlp
import os

def download_youtube_video(url: str, output_path: str = "data/input_videos/current_video.mp4") -> str:
    print(f"\n--- Downloading video from {url} ---")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # NEW: Delete the old video if it exists so we don't accidentally process it again!
    if os.path.exists(output_path):
        os.remove(output_path)
        print(" Deleted old cached video.")
    
    ydl_opts = {
        'format': 'best', 
        'outtmpl': output_path,
        'quiet': False,       
        'no_warnings': False,
        'noplaylist': True   
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(" Download complete!")
        return output_path
    except Exception as e:
        print(f"\n[YT-DLP ERROR]: {e}")
        return ""