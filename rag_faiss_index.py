import json
import os

import openai
import faiss
import numpy as np
from dotenv import load_dotenv
# 1. Set your OpenAI API key here or via environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Optionally, you can hardcode: openai.api_key = "sk-..."
def load_json(file_path):
    """Load the JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def json_to_chunks(data, parent_key=""):
    """
    Recursively traverse the JSON structure and create (text, metadata) pairs.
    """
    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            # Recurse
            chunks.extend(json_to_chunks(value, parent_key=new_key))

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_key = f"{parent_key}[{idx}]"
            chunks.extend(json_to_chunks(item, parent_key=new_key))

    else:
        # Base case: data is a string or other primitive, create a chunk
        text_chunk = f"{parent_key}: {data}"
        metadata = {"source": parent_key}
        chunks.append((text_chunk, metadata))

    return chunks

def get_embedding(text):
    """
    Get the embedding vector for a given piece of text using OpenAI's model.
    Adjust 'EMBEDDING_MODEL' as needed.
    """
    EMBEDDING_MODEL = "text-embedding-ada-002"  # 1536 dimensions typically

    response = openai.Embedding.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding, dtype=np.float32)

def create_faiss_index(embeddings):
    """
    Given a list of embedding vectors, create and return a FAISS index (L2 similarity).
    """
    # Determine dimensionality from the first embedding
    dimension = len(embeddings[0])
    
    # Create a flat (brute-force) index
    index = faiss.IndexFlatL2(dimension)

    # Convert embeddings to float32 NumPy array for indexing
    embeddings_array = np.array(embeddings, dtype=np.float32)
    index.add(embeddings_array)
    return index

def main():
    # --------------------------------------
    # 1. Load JSON from file
    # --------------------------------------
    # Path to your JSON (relative path from the script's directory)
    file_path = os.path.join("knowledge_base", "knowledge_base.json")
    data = load_json(file_path)

    # --------------------------------------
    # 2. Convert JSON to textual chunks
    # --------------------------------------
    chunks = json_to_chunks(data)
    # chunks is a list of tuples: [(text_chunk, metadata), (text_chunk, metadata), ...]

    # --------------------------------------
    # 3. Generate embeddings for each chunk
    # --------------------------------------
    texts = [chunk[0] for chunk in chunks]
    embeddings = [get_embedding(text) for text in texts]

    # --------------------------------------
    # 4. Create FAISS index
    # --------------------------------------
    index = create_faiss_index(embeddings)

    # --------------------------------------
    # 5. Save the index & metadata
    # --------------------------------------
    # We'll store them in the knowledge_base folder
    index_path = os.path.join("knowledge_base", "rag_faiss.index")
    faiss.write_index(index, index_path)

    metadata_list = [chunk[1] for chunk in chunks]
    metadata_path = os.path.join("knowledge_base", "rag_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print("FAISS index and metadata saved in 'knowledge_base' folder.")

if __name__ == "__main__":
    main()
