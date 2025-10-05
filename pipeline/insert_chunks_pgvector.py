def insert_chunks_into_pgvector(conn, all_chunks, chunk_embeddings, table_name="pdf_chunks_768"):
    """
    Insert chunks + embeddings into Postgres pgvector table.
    """
    
    cursor = conn.cursor()

    for i, chunk in enumerate(all_chunks):
        cursor.execute(
            f"""
            INSERT INTO {table_name}
            (id, content, embedding, report_date, page_num, type, placeholder)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                chunk["chunk_id"],
                chunk["chunk_text"],
                chunk_embeddings[i],
                chunk.get("report_date"),
                chunk.get("page_num"),
                chunk.get("type"),
                chunk.get("placeholder"),
            ),
        )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Inserted {len(all_chunks)} chunks into {table_name}")