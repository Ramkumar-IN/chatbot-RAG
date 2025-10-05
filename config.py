import os
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel
from doclayout_yolo import YOLOv10

# Load environment variables from .env
load_dotenv()

# Allow the specific class for safe unpickling
torch.serialization.add_safe_globals([ClassificationModel])

PDF_FOLDER = 'input_pdfs'
LLAMA_JSON_PATH = 'llamajson/combined_images_data.json'
# -------------------------------
# Database (pgvector) connection
# -------------------------------
PGVECTOR_HOST = "postgres-vectorstore.clmnkejpqspy.us-east-1.rds.amazonaws.com"
PGVECTOR_PORT = 5432
PGVECTOR_DB = "postgres"
PGVECTOR_USER = "postgres"

# Fetch password from secrets manager / userdata
# Example: from userdata.get("postgres_rws")
# Don’t hardcode it here!
PGVECTOR_PASSWORD_KEY = os.environ.get("PGVECTOR_PASSWORD_KEY")

if not PGVECTOR_PASSWORD_KEY:
    raise ValueError("Postgres password not set! Please check your .env file or environment variables.")

# Detection & Classification
model_detect = YOLOv10("models/doclayout_yolo_docstructbench_imgsz1024.pt")
model_classify = YOLO("models/classification_chart.pt")
