import os
from src.extract import extract_frames
from src.vlm import generate_frame_descriptions
from src.vector_db import store_frames_in_db, search_video

VIDEO_PATH = "data/input_videos/sample.mp4" 
FRAMES_DIR = "data/extracted_frames/"

def main():
    print("--- Starting Video Search System ---")
    
    try:
        # Phase 1: Extract Frames
        print("\n--- PHASE 1: Extraction ---")
        metadata = extract_frames(VIDEO_PATH, FRAMES_DIR, interval_seconds=5)
        
        # Phase 2: Generate Descriptions using Gemini
        if metadata:
            print("\n--- PHASE 2: VLM Processing ---")
            metadata_with_text = generate_frame_descriptions(metadata)
            
            # Phase 3: Store in Vector DB
            store_frames_in_db(metadata_with_text)
            
            # Phase 4: The Magic (Semantic Search)
            # Try a search that DOES NOT use the exact words from the description!
            search_query = "someone learning geometry"
            search_video(search_query, n_results=1)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()