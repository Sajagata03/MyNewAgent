import streamlit as st
import os
from dotenv import load_dotenv  


load_dotenv()

from src.download import download_youtube_video
from src.extract import extract_frames
from src.vlm import generate_frame_descriptions
from src.vector_db import store_frames_in_db, search_video, clear_database
#from src.audio import process_audio

# Initialize states
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
    st.session_state.has_audio = False  #  Memory for audio status

# 1. Page Configuration
st.set_page_config(page_title="VisionSearch AI", layout="wide")

# 2. Custom CSS Injection 
st.markdown("""
    <style>
    /* Clean up the main background and buttons */
    .stButton>button { border-radius: 8px; font-weight: bold; }
    /* Ensure images never break out of their containers */
    img { max-width: 100%; border-radius: 8px; }
    /* Style the AI reasoning boxes */
    .stAlert { border-radius: 8px; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)



# Initialize states
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# ==========================================
# POP-UP MODAL (Triggers when processing is done)
# ==========================================
@st.dialog(" AI Engine Initialization Complete")
def show_summary_popup(metadata):
    st.success("Video successfully embedded into the Vector Database!")
    st.markdown("### What the AI sees:")
    st.markdown("Review the scenes the AI extracted before you search.")
    
    for frame in metadata:
        if frame.get("description"):
            st.markdown(f"** `{frame['timestamp_formatted']}`**")
            st.caption(frame['description'])
            st.markdown("---")
            
    if st.button("Close & Start Searching", type="primary", use_container_width=True):
        st.rerun()

# ==========================================
# SIDEBAR: INGESTION PIPELINE (VISION ONLY)
# ==========================================
with st.sidebar:
    st.title(" Engine Control")
    st.markdown("Ingest new video data into the Vector Database.")
    st.markdown("---")
    
    video_url = st.text_input("Video URL", placeholder="Paste any video link here...")
    
    if st.button("Process & Index Video", use_container_width=True):
        if video_url:
            with st.status("Initializing AI Pipeline...", expanded=True) as status:
                try:
                    st.write(" Downloading video...")
                    video_path = download_youtube_video(video_url, "data/input_videos/current_video.mp4")
                    if not video_path: st.stop()

                    st.write(" Extracting frames...")
                    metadata = extract_frames(video_path, "data/extracted_frames", interval_seconds=4) 

                    st.write(" Vision Language Model analyzing frames...")
                    try:
                        metadata_with_text = generate_frame_descriptions(metadata)
                    except Exception as vlm_err:
                        st.error(f"VLM Error: {vlm_err}")
                        st.stop()

                    valid_frames = [f for f in metadata_with_text if f.get("description", "") != ""]
                    if len(valid_frames) == 0:
                        status.update(label="AI Processing Failed.", state="error")
                        st.stop()

                    st.write(" Wiping old memories and embedding Vision Data...")
                    clear_database() 
                    store_frames_in_db(metadata_with_text) 

                    status.update(label="System Ready!", state="complete", expanded=False)
                    
                    # Force the UI to know we are in Vision-Only mode
                    st.session_state.video_processed = True
                    st.session_state.has_audio = False 
                    
                    show_summary_popup(metadata_with_text)

                except Exception as e:
                    status.update(label="Pipeline Failure.", state="error")
                    st.error(str(e))
        else:
            st.warning("Provide a valid URL.")

# ==========================================
# MAIN DASHBOARD: SEARCH
# ==========================================
st.title("VisionSearch AI ")
st.markdown("Search inside video content using pure natural language.")

if not st.session_state.video_processed:
    st.info("👈 Please process a video in the sidebar to begin searching.")
else:
    st.markdown("---")
    
    search_col, setting_col = st.columns([3, 1])
    with search_col:
        query = st.text_input("What are you looking for?", placeholder="e.g., A brown bird")
    with setting_col:
        num_results = st.number_input("Top Matches", min_value=1, max_value=6, value=2)

    if st.button(" Search Video", type="primary"):
        if query:
            with st.spinner("Scanning database..."):
                results = search_video(query, n_results=num_results)
                
                if results:
                    st.success(f"Top Matches for '{query}'")
                    for i, match in enumerate(results):
                        st.markdown(f"### Match 0{i+1} —  `{match['timestamp_formatted']}`")
                        
                        col_img, col_data = st.columns([1.2, 1]) 
                        
                        with col_img: 
                            st.image(match["frame_path"], width=450) 
                            
                        with col_data:
                            st.markdown("**AI Context Reasoning:**")
                            # This box will now show the visual description AND the spoken text!
                            st.info(match["description"]) 
                            
                        st.markdown("---")
                else:
                    st.error("No results found.")