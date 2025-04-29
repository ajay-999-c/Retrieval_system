import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os
import time

# Start timing
start_time = time.time()
# 1. Load the tagged FAQ CSV
df = pd.read_csv("data/q_and_a_dataset.csv")
print(f"✅ Loaded {len(df)} FAQs.")

# 2. Prepare Chunks and Metadata
texts = []
metadatas = []

for idx, row in df.iterrows():
    question = str(row['Question']).strip()
    reply = str(row['Reply']).strip()
    tag = str(row['Tagging']).strip()

    combined_text = f"Question: {question}\nAnswer: {reply}"
    texts.append(combined_text)

    metadata = {
        "section": tag,
        "question": question
    }
    metadatas.append(metadata)

print(f"✅ Prepared {len(texts)} chunks and metadata.")

# 3. Generate Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ HuggingFace embedding model loaded.")

# 4. Create Chroma Vectorstore
persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=persist_directory
)

end_time = time.time()
print(f"🎯 Vectorstore created successfully! Total time: {end_time - start_time:.2f} seconds.")
print(f"📂 Chroma DB saved at: {persist_directory}")

