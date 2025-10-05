from collections import deque
from typing import List
from google import genai

from pipeline.embeddings import get_google_embeddings_raw

class ChatbotWrapper:
    def __init__(self, run_pipeline_func, conn, llamageneratedjson, client: genai.Client, max_history=100):
        """
        Interactive chatbot wrapper with memory and LLM follow-up handling.

        Args:
            run_pipeline_func: Your retrieval pipeline function
                               (run_query_pipeline(query_text, query_embedding, conn, llamageneratedjson))
            conn: Active Postgres connection
            llamageneratedjson: Pre-loaded llama JSON for figures
            client: Gemini LLM client
            max_history: Number of past conversations to keep
        """
        self.run_pipeline_func = run_pipeline_func
        self.conn = conn
        self.llamageneratedjson = llamageneratedjson
        self.client = client
        self.max_history = max_history
        self.history = deque(maxlen=max_history)  # stores (user, bot) tuples

    # -----------------------------
    # LLM-based follow-up answer
    # -----------------------------
    def answer_with_history(self, user_input: str) -> str | None:
        """
        Sends user input + last conversation history to LLM.
        Returns LLM answer, or None if LLM call fails.
        """
        # Prepare context
        history_text = ""
        for u, b in list(self.history)[-self.max_history:]:
            history_text += f"User: {u}\nBot: {b}\n"

        prompt = f"""
You are a helpful assistant chatbot. Use the following conversation history to answer the user's new question.
If the question is general chit-chat, respond normally.
If the answer is not available from previous conversation or requires new data, respond: "I don't have enough information, please run a new query."

Conversation history:
{history_text}

New Question: {user_input}

Answer:
"""
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer_text = resp.text.strip()
        except Exception as e:
            print("⚠️ LLM call failed:", e)
            return None

        return answer_text

    # -----------------------------
    # Retrieval pipeline fallback
    # -----------------------------
    def run_new_query(self, user_input: str) -> str:
        """
        Calls the full retrieval pipeline using embeddings + Postgres + Gemini.
        """
        query_embedding = get_google_embeddings_raw(self.client, [user_input])[0]  # list of floats
        answer = self.run_pipeline_func(user_input, query_embedding, self.conn, self.llamageneratedjson, self.client)
        return answer

    # -----------------------------
    # Main entry point
    # -----------------------------
    def respond(self, user_input: str) -> str:
        """
        Handles user input:
        - Always sends input + last conversation history to LLM
        - LLM decides if it can answer (chit-chat) or needs a pipeline run
        - If a query is needed, politely asks user for confirmation before running
        """

        # ------------------------------
        # 1️⃣ Prepare history context for LLM
        # ------------------------------
        history_text = ""
        for user_msg, bot_msg in list(self.history)[-self.max_history:]:
            history_text += f"User: {user_msg}\nBot: {bot_msg}\n"

        full_prompt = history_text + f"User: {user_input}\nBot:"

        # ------------------------------
        # 2️⃣ Get LLM answer using history
        # ------------------------------
        llm_answer = self.answer_with_history(full_prompt)

        if llm_answer:
            self.history.append((user_input, llm_answer))

            # ------------------------------
            # 3️⃣ Check if LLM cannot answer (requires new data)
            # ------------------------------
            if "don't have enough information" in llm_answer.lower() or \
               "cannot answer" in llm_answer.lower():

                professional_msg = (
                    "I am an AI assistant specialized in helping you query and explore your documents. "
                    "For questions outside general conversation, I can fetch the latest data. "
                    "Do you want me to run a query for this?"
                )
                print(professional_msg)
                choice = input("Please type 'y' to run the query, or 'n' to continue chit-chat: ").strip().lower()

                if choice == "y":
                    query_text = input("Please type the query you want me to run: ").strip()
                    pipeline_answer = self.run_new_query(query_text)
                    self.history.append((query_text, pipeline_answer))
                    return pipeline_answer
                else:
                    fallback = "Let's continue the conversation. Ask me something else!"
                    self.history.append((user_input, fallback))
                    return fallback

            # ------------------------------
            # 4️⃣ Normal LLM answer (chit-chat or answer from history)
            # ------------------------------
            return llm_answer

        # ------------------------------
        # 5️⃣ LLM call failed completely
        # ------------------------------
        fallback = "Sorry, I cannot process your request right now."
        self.history.append((user_input, fallback))
        return fallback

    # -----------------------------
    # Utility
    # -----------------------------
    def print_history(self):
        """Debug: print last N conversations"""
        for i, (u, b) in enumerate(self.history, 1):
            print(f"{i}. User: {u}\n   Bot: {b}\n")
