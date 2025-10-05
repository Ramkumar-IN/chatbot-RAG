import json
def load_llama_json(llama_json_path):
    """Load llama-generated JSON from disk."""
    with open(llama_json_path, "r") as f:
        llamageneratedjson = json.load(f)
    return llamageneratedjson

def normalize_llama_placeholders(llamageneratedjson):
    """
    Normalize placeholder field inside llama-generated JSON.
    Removes `.png`, trims whitespace.
    """
    for node in llamageneratedjson:
        if "placeholder" in node and isinstance(node["placeholder"], str):
            node["placeholder"] = node["placeholder"].replace(".png", "").strip()

    print("✅ Normalized Llama JSON placeholders")
    return llamageneratedjson