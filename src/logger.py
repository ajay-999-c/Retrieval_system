import logging
import os
from datetime import datetime
import csv
import json

# 1. Setup log directories if not exist
if not os.path.exists("logs"):
    os.makedirs("logs")

if not os.path.exists("full_logs"):
    os.makedirs("full_logs")

# 2. Setup Logger Instances
rag_logger = logging.getLogger("rag_pipeline")
rag_logger.setLevel(logging.INFO)
rag_file_handler = logging.FileHandler("logs/rag_pipeline.log", encoding="utf-8")
rag_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
rag_file_handler.setFormatter(rag_formatter)
rag_logger.addHandler(rag_file_handler)

embedding_logger = logging.getLogger("embedding_creation")
embedding_logger.setLevel(logging.INFO)
embedding_file_handler = logging.FileHandler("logs/embedding_creation.log", encoding="utf-8")
embedding_file_handler.setFormatter(rag_formatter)
embedding_logger.addHandler(embedding_file_handler)

# --- LOGGING FUNCTIONS ---
def log_retrieval_results(docs, log_file="logs/retrieved_results.csv"):
    """
    Save retrieved document results (Section, Question, Answer) into a CSV log file.

    Args:
        docs (List[Document]): Retrieved LangChain documents.
        log_file (str): Output CSV file path.
    """
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Timestamp", "Result No", "Section", "Original Question", "Answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for idx, doc in enumerate(docs, 1):
            writer.writerow({
                "Timestamp": timestamp,
                "Result No": idx,
                "Section": doc.metadata.get("section", "Unknown"),
                "Original Question": doc.metadata.get("question", "Unknown"),
                "Answer": doc.page_content
            })

    print(f"✅ Retrieved results saved to: {log_file}")


def log_event(message: str):
    """
    Log a generic event or step inside the RAG pipeline system.
    Appends the message to logs/rag_pipeline.log.
    """
    rag_logger.info(message)

def log_pipeline_step(step_name, input_text, input_tokens, output_tokens, time_taken, section_type=None, retrieval_size=None, user_id=None):
    """
    Log a detailed structured step inside the pipeline, mentioning tokens, retrieval size, timing, etc.
    Appends info into logs/rag_pipeline.log.
    """
    log_message = f"USER: {user_id} | STEP: {step_name} | INPUT TOKENS: {input_tokens} | OUTPUT TOKENS: {output_tokens} | TIME: {time_taken:.2f}s | SECTION: {section_type} | RETRIEVED DOCS: {retrieval_size} | INPUT: {input_text}"
    rag_logger.info(log_message)

def save_full_pipeline_log(log_data: dict, user_id: str):
    """
    Save full detailed RAG pipeline log for one user interaction.
    Appends into:
      - full_logs/full_pipeline_log.csv (all sessions together)
      - full_logs/full_pipeline_log.json (full session dump)
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- 1. Append to master CSV file ---
    csv_path = "full_logs/full_pipeline_log.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "Timestamp", "User ID", "Question",
            "Rewritten Query", "Sub-Questions",
            "Chunks Retrieved", "Prompt Context",
            "Generated Answer", "Input Tokens", "Output Tokens", "Total Time (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Timestamp": timestamp,
            "User ID": user_id,
            "Question": log_data["user_query"],
            "Rewritten Query": log_data["rewritten_query"],
            "Sub-Questions": " | ".join(log_data["sub_questions"]),
            "Chunks Retrieved": len(log_data["retrieved_chunks"]),
            "Prompt Context": log_data["final_prompt"][:500],
            "Generated Answer": log_data["generated_answer"],
            "Input Tokens": log_data["input_tokens"],
            "Output Tokens": log_data["output_tokens"],
            "Total Time (s)": f"{log_data['total_time']:.2f}"
        })

    print(f"✅ Full CSV log updated at {csv_path}")

    # --- 2. Append to master JSON file ---
    json_path = "full_logs/full_pipeline_log.json"
    session_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "user_query": log_data["user_query"],
        "rewritten_query": log_data["rewritten_query"],
        "sub_questions": log_data["sub_questions"],
        "retrieved_chunks": log_data["retrieved_chunks"],
        "final_prompt": log_data["final_prompt"],
        "generated_answer": log_data["generated_answer"],
        "input_tokens": log_data["input_tokens"],
        "output_tokens": log_data["output_tokens"],
        "total_time_seconds": log_data["total_time"],
        "section_type": log_data.get("section_type", "unknown")
    }

    if not os.path.isfile(json_path):
        all_sessions = []
    else:
        with open(json_path, "r", encoding="utf-8") as jf:
            try:
                all_sessions = json.load(jf)
            except json.decoder.JSONDecodeError:
                all_sessions = []

    all_sessions.append(session_entry)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(all_sessions, jf, indent=2)

    print(f"✅ Full JSON log updated at {json_path}")

def log_embedding_event(message: str):
    """
    Log an embedding/vectorstore creation event.
    Appends the message to logs/embedding_creation.log.
    """
    embedding_logger.info(message)
