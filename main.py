"""
Main entry point for Chatbot RAG pipeline
"""

from pipeline.chatbot import ChatbotWrapper
from pipeline.database import connect_pg
from pipeline.embeddings import get_google_embeddings_raw
from pipeline.insert_chunks_pgvector import insert_chunks_into_pgvector
from pipeline.llamajson import process_llama_json
from pipeline.pdf_processing import extract_text_and_chunks
from pipeline.run_query import run_query
from pipeline.summarization import summarize_text

# Configs
from config import PDF_FOLDER, LLAMA_JSON_PATH


def main():
    print("🚀 Starting Chatbot RAG pipeline...")

    # 1. Process PDF and extract chunks
    print("📄 Extracting text + chunks from PDF...")
    all_chunks = extract_text_and_chunks(PDF_FOLDER)

    # 2. Get embeddings for chunks
    print("🧠 Generating embeddings...")
    chunk_embeddings = get_google_embeddings_raw([c["chunk_text"] for c in all_chunks])

    # 3. Connect to database
    print("🔌 Connecting to Postgres...")
    conn = connect_pg()

    # 4. Insert into pgvector
    print("💾 Inserting chunks into pgvector...")
    insert_chunks_into_pgvector(conn, all_chunks, chunk_embeddings)

    # 5. Optional: Summarize
    print("📝 Summarizing content...")
    summary = summarize_text(" ".join(c["chunk_text"] for c in all_chunks))
    print("\n=== SUMMARY ===\n", summary[:1000], "...\n")

    # 6. Run a sample query
    print("🔍 Running test query...")
    result = run_query("Give me insights from the document", conn)
    print("\n=== QUERY RESULT ===\n", result, "\n")

    # 7. Chatbot ready
    print("🤖 Chatbot is ready! You can now interact with it.")
    bot = ChatbotWrapper(conn)
    print(bot.ask("Hello!"))


if __name__ == "__main__":
    main()


