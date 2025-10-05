# ===============================
# 5️⃣ Split into chunks (in-memory)
# ===============================
from google.genai import types
import numpy as np  
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_full_json(full_json_text_table_charts, chunk_size=2500, chunk_overlap=300):
    all_chunks=[]
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    placeholder_pattern=re.compile(r"^\[.*?_(Chart|Table)\d+_Page\d+\]$")
    for node in full_json_text_table_charts:
        combined_text=node.get("text","")
        for key,value in node.items():
            if isinstance(value,str) and placeholder_pattern.match(key):
                combined_text+="\n\n"+value
        chunks=text_splitter.split_text(combined_text)
        for i,chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id":f"{node.get('id')}_chunk{i}",
                "chunk_text":chunk_text,
                "placeholder":node.get("placeholder",[]),
                "page_num":node.get("page_num"),
                "type":node.get("node-type"),
                "report_date":node.get("report_date")


            })
    return all_chunks

def get_google_embeddings_raw(client, chunk_texts, batch_size=100, dim=768):
    """
    Generate raw embeddings for chunk_texts using Gemini embedding API
    without any normalization.

    Args:
        chunk_texts (list[str]): List of texts to embed.
        batch_size (int): Max 100, API limitation.
        dim (int): Desired embedding dimensionality (default 768).

    Returns:
        list[list[float]]: List of raw embeddings (each a list of floats).
    """
    all_embeddings = []

    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=dim)
        )

        for emb in result.embeddings:
            # directly convert to float32 without normalization
            vec = np.array(emb.values, dtype=np.float32)
            all_embeddings.append(vec.tolist())

    return all_embeddings