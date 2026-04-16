import chromadb
from typing import List, Dict

# Initialize ChromaDB client. 
# This creates a database folder inside your data directory.
client = chromadb.PersistentClient(path="./data/chroma_db")

# Create or load a collection (think of this like a table in SQL)
collection = client.get_or_create_collection(name="video_frames")

def store_frames_in_db(frame_metadata: List[Dict]):
    """Stores the text descriptions and metadata into the Vector DB."""
    print("\n--- PHASE 3: Vector Storage ---")
    
    documents = []
    metadatas = []
    ids = []
    
    for i, data in enumerate(frame_metadata):
        description = data.get("description", "")
        
        # Skip any frames that had API errors
        if "Error" in description or not description:
            continue
            
        documents.append(description)
        # We store the path and timestamp so we can show them to the user later
        metadatas.append({
            "timestamp_formatted": data["timestamp_formatted"],
            "frame_path": data["frame_path"]
        })
        ids.append(f"frame_{i}_{data['timestamp_formatted']}")
        
    if documents:
        # This step automatically converts the text to vectors using a built-in model!
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully saved {len(documents)} frames to ChromaDB!")
    else:
        print("No valid descriptions to store.")

def search_video(query_text: str, n_results: int = 3):
    """Searches the database and returns the top visual matches."""
    # We query the database, asking for the top N results
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    matches = []
    # Package up all the results
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            matches.append({
                "description": results['documents'][0][i],
                "timestamp_formatted": results['metadatas'][0][i]['timestamp_formatted'],
                "frame_path": results['metadatas'][0][i]['frame_path'],
                # Distance: Lower is better (closer match)
                "distance": results['distances'][0][i] 
            })
    return matches

    # Format the output nicely
    if results['documents'][0]:
        for i in range(len(results['documents'][0])):
            desc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            
            print(f"\nMatch Found!")
            print(f"Timestamp: {meta['timestamp_formatted']}")
            print(f"Image File: {meta['frame_path']}")
            print(f"AI Description: {desc}")
    else:
        print("No matches found.")


def clear_database():
    """Wipes the database clean for a new video."""
    try:
        client.delete_collection(name="video_frames")
        global collection
        collection = client.get_or_create_collection(name="video_frames")
    except Exception:
        pass