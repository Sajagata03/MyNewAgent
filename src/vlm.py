import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from PIL import Image
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def generate_frame_descriptions(frame_metadata: List[Dict]) -> List[Dict]:
    print(f"\n Starting VLM processing for {len(frame_metadata)} frames...")
    
    for i, frame_data in enumerate(frame_metadata):
        image_path = frame_data["frame_path"]
        prompt = (
            "Describe exactly what is happening in this video frame in 1 clear sentence. "
            "Focus on the main subject, colors, and action. "
            "IMPORTANT: On a new line, add a comma-separated list of 5 simple keywords "
            "that include the primary color and the generic name of the object. "
            "Example: 'Keywords: brown, bird, animal, tree, outdoors'"
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = Image.open(image_path)
                
                
                response = client.models.generate_content(
                    
                    model='gemini-2.5-flash-lite', 
                    contents=[prompt, img]
                )
                
                description = response.text.strip()
                frame_data["description"] = description
                print(f" Frame [{frame_data['timestamp_formatted']}] processed: {description[:40]}...")
                
                # The 6-second safety buffer to guarantee we stay under rate limits
                time.sleep(15) 
                break 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    wait_time = 5 * (attempt + 1)
                    print(f" Rate Limit Hit. Fast-pausing for {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f" Error on frame [{frame_data['timestamp_formatted']}]: {error_msg}")
                    frame_data["description"] = ""
                    break 
            
    print("VLM Processing Complete!")
    return frame_metadata