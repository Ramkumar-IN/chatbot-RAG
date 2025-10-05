
import re  
import dateparser
import json

# --------------------------
# 2️⃣ Extract report dates
# --------------------------
def extract_report_dates(query: str):
    dates = []
    matches = re.findall(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s?['\-]?\s?(\d{2,4})",
        query, re.I
    )
    for month, year in matches:
        year = "20" + year if len(year) == 2 else year
        parsed = dateparser.parse(f"{month} {year}")
        if parsed:
            dates.append(parsed.strftime("%Y-%m"))
    qmatches = re.findall(r"C([1-4])Q(\d{2})", query, re.I)
    for quarter, year in qmatches:
        year = "20" + year
        dates.append(f"{year}-Q{quarter}")
    return list(set(dates))

# --------------------------
# 3️⃣ Extract keywords via Gemini
# --------------------------
def extract_doc_keywords_gemini(query: str):
    prompt = f"""
    Extract the company or document names mentioned in this query.
    Return only the names as a comma-separated list.
    Example: 'Edgewater and TrendForce data' -> Edgewater, TrendForce
    Query: {query}
    """
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text_response = resp.text.strip()
        if not text_response:
            raise ValueError("Gemini returned empty text")
    except Exception:
        text_response = query
        print("⚠️ Gemini returned no text, fallback: using query text")
    keywords = [kw.strip() for kw in re.split(r",|\n", text_response) if kw.strip()]
    return list(set(keywords))

# --------------------------
# 4️⃣ Retrieve document IDs
# --------------------------
def get_available_ids(cursor):
    cursor.execute("SELECT DISTINCT id FROM pdf_chunks_768;")
    return [row[0] for row in cursor.fetchall()]

# --------------------------
# 5️⃣ Retrieve chunks by similarity with interactive fallback
# --------------------------
def retrieve_chunks(cursor, query_embedding, report_dates, matched_docs, similarity_threshold=0.5):
    query_str = "[" + ",".join(map(str, query_embedding)) + "]"
    sql_similarity = """
    SELECT id, content, report_date, placeholder,
           1 - (embedding <-> %s) AS similarity
    FROM pdf_chunks_768
    ORDER BY embedding <-> %s;
    """
    cursor.execute(sql_similarity, (query_str, query_str))
    all_chunks = cursor.fetchall()

    # Filter by explicit report_dates
    explicit_dates = [d for d in report_dates if d != "LATEST"]
    if explicit_dates:
        date_filtered_chunks = [row for row in all_chunks if row[2] in explicit_dates]
    unique_dates = sorted(set(row[2] for row in date_filtered_chunks))
    print("✅ Unique report dates in filtered chunks:", unique_dates)

    # Filter by LATEST
    if "LATEST" in report_dates:
        latest_dates = [row[2] for row in all_chunks if row[2] is not None]
        if latest_dates:
            latest_date = max(latest_dates)
            date_filtered_chunks = [row for row in all_chunks if row[2] == latest_date]

    # Filter by matched docs
    if matched_docs:
        doc_filtered_chunks = [row for row in date_filtered_chunks if row[0] in matched_docs]
    else:
      doc_filtered_chunks = date_filtered_chunks
    print(f"   ↳ Retrieved {len(doc_filtered_chunks)} chunks (after date filtering)")

    filtered_chunks = [row for row in doc_filtered_chunks if row[4] >= similarity_threshold]

    # Interactive fallback
    if not filtered_chunks and doc_filtered_chunks:
        best_row = max(doc_filtered_chunks, key=lambda x: x[4])
        lowest_cosine = best_row[4]
        print(f"⚠️ No chunks meet threshold {similarity_threshold}.")
        print(f"   ↳ Highest available similarity = {lowest_cosine:.4f}, from doc_id={best_row[0]}")
        user_input = input(
            f"Do you want to use this chunk? (y/n) or specify a new minimum cosine value: "
        ).strip().lower()
        if user_input == "y":
            filtered_chunks = [best_row]
            similarity_threshold = lowest_cosine
        else:
            try:
                new_threshold = float(user_input)
                similarity_threshold = new_threshold
                filtered_chunks = [row for row in doc_filtered_chunks if row[4] >= new_threshold]
            except ValueError:
                filtered_chunks = []

    print(f"   ↳ Retrieved {len(filtered_chunks)} chunks (after cosine ≥ {similarity_threshold})")
    return filtered_chunks



# --------------------------
# 6️⃣ Match figures from llamageneratedjson
# --------------------------
def match_figures(retrieved_chunks, llamageneratedjson):
    # Split multiple placeholders
    relevant_placeholders = []
    for chunk in retrieved_chunks:
        ph_str = chunk.get("placeholder")
        if ph_str:
            # Remove curly braces and split by comma
            ph_str = ph_str.strip("{}")
            ph_list = [p.strip() for p in ph_str.split(",") if p.strip()]
            relevant_placeholders.extend(ph_list)

    # Now match each placeholder individually
    matching_nodes = [node for node in llamageneratedjson if node["placeholder"] in relevant_placeholders]
    print(f"   ↳ Found {len(matching_nodes)} matching nodes in llamageneratedjson")

    figures_to_pass = []
    for node in matching_nodes:
        for fig in node.get("figures", []):
            fig_copy = fig.copy() if isinstance(fig, dict) else {"text": fig}
            fig_copy["placeholder"] = node.get("placeholder", "Unknown")
            figures_to_pass.append(fig_copy)
    #figures_to_pass = figures_to_pass[:2]
    print(f"   ↳ Passing {len(figures_to_pass)} figures to LLM")
    return figures_to_pass

# --------------------------
# 7️⃣ Build LLM prompt
# --------------------------
def build_prompt(query_text, retrieved_chunks, figures_to_pass):
    prompt = f"""
You are given:

1️⃣ Retrieved text chunks, with source and report_date metadata.
2️⃣ Figures in JSON format, each with its original placeholder.

Task:
For Retrieved text chunks:
- Summarize the key points from the text chunks.
- Use tables wherever possible to provide an easy understandable view.
- Display the document source
For Figures in JSON format:
- Convert figures into human-readable tables with rows/columns.
- Extract 'document source' from placeholder by stripping '_Chart/Table_PageXX'.
- Show 'report_date' alongside the document source.
- Provide short insights for each figure.
- Use only relevant figures that matches with the retrieved text chunks
- Use markdown formatting.
- Important is always have year or quarter or any time period as column headers.

=== Retrieved Chunks ===
{json.dumps(retrieved_chunks, indent=2)}

=== Figures JSON ===
{json.dumps(figures_to_pass, indent=2)}

Question: {query_text}
"""
    print(f"   ↳ Prompt length: {len(prompt)} chars")
    return prompt

# --------------------------
# 8️⃣ Send prompt to Gemini
# --------------------------
def send_to_gemini(prompt, client):
    print("🔹 Step 10: Sending to Gemini...")
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    answer = resp.text.strip()
    print("   ↳ Gemini response received!")
    return answer

# --------------------------
# 9️⃣ Main pipeline
# --------------------------
def run_query_pipeline(query_text, query_embedding,conn, llamageneratedjson,client):

    cursor = conn.cursor()

    print("🔹 Step 2: Extracting report dates...")
    report_dates = extract_report_dates(query_text)
    if not report_dates:
        report_dates = ["LATEST"]
    print(f"   ↳ Extracted report_dates: {report_dates}")

    print("🔹 Step 4: Extracting keywords via Gemini...")
    keywords = extract_doc_keywords_gemini(query_text)

    print("🔹 Step 5: Loading available document IDs...")
    available_ids = get_available_ids(cursor)
    matched_docs = []
    for kw in keywords:
        for original_id in available_ids:
            if kw.lower() in original_id.lower():
                matched_docs.append(original_id)
    print(f"   ↳ Extracted keywords: {keywords}")
    print(f"   ↳ Matched docs: {matched_docs}")

    retrieved_chunks = retrieve_chunks(cursor, query_embedding, report_dates, matched_docs)
    print(retrieved_chunks)
    retrieved_chunks_with_sources = [
        {"id": row[0], "content": row[1], "report_date": row[2], "placeholder": row[3], "similarity": row[4]}
        for row in retrieved_chunks
    ]

    figures_to_pass = match_figures(retrieved_chunks_with_sources, llamageneratedjson)
    prompt = build_prompt(query_text, retrieved_chunks_with_sources, figures_to_pass)
    answer = send_to_gemini(prompt, client)

    print("\n=== Question ===")
    print(query_text)
    print("\n=== Answer ===")
    print(answer)
    return answer
