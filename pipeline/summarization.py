from PIL import Image
import cv2
def summarize_image(client,img, prompt):
    """
    Core Gemini summarization function.
    img: PIL Image
    prompt: str
    Returns summarized text
    """
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]  # list of str + image
    )
    text = " ".join(
        line.strip().replace("|", " ")
        for line in resp.text.strip().split("\n")
        if line.strip()
    )
    return text

def summarize_all_table_chart_nodes_in_memory(client,table_chart_nodes):
    """
    Summarizes all in-memory chart/table nodes using Gemini.

    table_chart_nodes: list of dicts, each dict has keys:
        - "img": numpy array (BGR from cv2)
        - "placeholder": string
        - "page_num": int
        - "node-type": "chart" or "table"

    """
    summarized_nodes = []

    for node in table_chart_nodes:
        # Choose prompt based on node type
        prompt = (
            "Summarize chart in 3-4 sentences: include title, axes, main trends."
            if node["node-type"] == "chart"
            else "Summarize table in 6-8 sentences including title, rows, columns, trends, highest/lowest values."
        )

        # Convert OpenCV BGR numpy array to PIL RGB image
        pil_img = Image.fromarray(cv2.cvtColor(node["img"], cv2.COLOR_BGR2RGB))

        # Call Gemini
        summary = summarize_image(client,pil_img, prompt)

        # Create a copy of the node with the summary
        node_copy = node.copy()
        node_copy["text"] = summary
        summarized_nodes.append(node_copy)

    return summarized_nodes

def merge_text_and_table_charts(all_text_json, summarized_table_chart_nodes):
    table_chart_lookup={n["placeholder"]:n["text"] for n in summarized_table_chart_nodes}
    full_json=[]
    for item in all_text_json:
        merged=item.copy()
        for ph in item.get("placeholder",[]):
            if ph in table_chart_lookup: merged[ph]=table_chart_lookup[ph]
        full_json.append(merged)
    return full_json
