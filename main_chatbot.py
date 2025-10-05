"""
Interactive Chatbot using PGVector + Gemini + llamajson
"""

from pipeline.chatbot import ChatbotWrapper
from pipeline.database import connect_pg
from pipeline.run_query import run_query_pipeline
from pipeline.llamajson import load_llama_json, normalize_llama_placeholders
from config import LLAMA_JSON_PATH

# -----------------------------
# 1️⃣ Connect to PGVector
# -----------------------------
print("🔌 Connecting to Postgres...")
conn = connect_pg()

# -----------------------------
# 2️⃣ Load llamajson
# -----------------------------
print("📄 Loading Llama JSON...")

llamageneratedjson = load_llama_json(LLAMA_JSON_PATH)
llamageneratedjson = normalize_llama_placeholders(llamageneratedjson)

# -----------------------------
# 3️⃣ Initialize Gemini client
# -----------------------------
from google import genai
client = genai.Client()

# -----------------------------
# 4️⃣ Initialize ChatbotWrapper
# -----------------------------
chatbot = ChatbotWrapper(
    run_pipeline_func=run_query_pipeline,  # full pipeline function
    conn=conn,                              # active Postgres connection
    llamageneratedjson=llamageneratedjson,  # preloaded llama JSON
    client=client,                          # Gemini client
    max_history=25                           # number of turns to remember
)

# -----------------------------
# 5️⃣ Commands and CLI help
# -----------------------------
COMMAND_BAR = "💡 Commands:  [exit] [quit] [history] [help] [query:<text>]"
COMMANDS = {
    "exit": "Quit the chatbot",
    "quit": "Quit the chatbot",
    "history": "Show recent conversation history",
    "help": "Show this list of commands",
    "query": "Run a document query (example: 'query: <your query>')"
}

print("💬 Chatbot ready! Type 'help' to see available commands. Type 'exit' to quit.")
print(COMMAND_BAR)

# -----------------------------
# 6️⃣ Interactive chat loop
# -----------------------------
while True:
    user_input = input("\nYou: ").strip()

    # Handle special commands first
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    elif user_input.lower() == "help":
        print("\n📜 Available commands:")
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd} - {desc}")
        continue
    elif user_input.lower() == "history":
        print("\n🕑 Conversation History:")
        chatbot.print_history()
        continue
    elif user_input.lower().startswith("query:"):
        query_text = user_input[len("query:"):].strip()
        answer = chatbot.run_new_query(query_text)
        print(f"\nBot: {answer}")
        continue

    # Default: normal chat response
    answer = chatbot.respond(user_input)
    print(f"\nBot: {answer}")

    # Reprint command bar below each response
    print("\n" + COMMAND_BAR)
