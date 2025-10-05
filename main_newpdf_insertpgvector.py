"""
Main pipeline: process PDFs, summarize, create embeddings, insert into PGVector
"""
import os
from pipeline.pdf_processing import process_pdfs
from pipeline.summarization import summarize_all_table_chart_nodes_in_memory, merge_text_and_table_charts
from pipeline.embeddings import chunk_full_json, get_google_embeddings_raw
from pipeline.database import connect_pg
from pipeline.insert_chunks_pgvector import insert_chunks_into_pgvector
from config import PDF_FOLDER

from google import genai
client = genai.Client()

print("hi")
def main():
    print("🚀 Running full PDF → PGVector pipeline...")

    # 1️⃣ Process PDFs
    all_placeholders, all_text_json, all_table_chart_nodes = process_pdfs(PDF_FOLDER)
    print(f"Processed {len(all_text_json)} pages, {len(all_table_chart_nodes)} table/chart nodes!")

    # 2️⃣ Summarize tables/charts in-memory
    summarized_table_chart_nodes = summarize_all_table_chart_nodes_in_memory(client,all_table_chart_nodes)
    print(f"Summarized {len(summarized_table_chart_nodes)} table/chart nodes!")

    # 3️⃣ Merge text and table/chart nodes
    full_json_text_table_charts = merge_text_and_table_charts(all_text_json, summarized_table_chart_nodes)
    print(f"Merged {len(all_text_json)} pages with {len(summarized_table_chart_nodes)} table/chart nodes!")

    # 4️⃣ Chunk full JSON
    all_chunks = chunk_full_json(full_json_text_table_charts)
    print(f"Processed {len(all_text_json)} pages, {len(summarized_table_chart_nodes)} table/chart nodes, {len(all_chunks)} total chunks in-memory!")

    # 5️⃣ Generate embeddings
    chunk_texts = [c['chunk_text'] for c in all_chunks]
    chunk_embeddings = get_google_embeddings_raw(client,chunk_texts)
    print(f"Generated {len(chunk_embeddings)} embeddings!")

    # 6️⃣ Insert chunks + embeddings into PGVector
    print("🔌 Connecting to Postgres...")
    conn = connect_pg()
    insert_chunks_into_pgvector(conn,all_chunks, chunk_embeddings)
    print(f"Inserted/Appended {len(all_chunks)} chunks into PGVector!")

    

if __name__ == "__main__":
    main()