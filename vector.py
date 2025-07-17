from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import numpy as np
import pandas as pd
import os
from langchain_core.documents  import Document
import tqdm

df = pd.read_csv("zomato.csv")

embeddings = OllamaEmbeddings(model="llama3.2")

db_path  = "./chroma_db"

check_db = not os.path.exists(db_path)


#Pre-process data
df = df.dropna()
df = df.reset_index(drop=True)
df = df.drop(df.columns[:2], axis=1)

if check_db:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content= row['restaurant name'] + " " + row['cuisines type'] + " " + row['area'],
            metadata={
                "Rating": row['rate (out of 5)'],
                "Avg Cost for two": row['avg cost (two people)'],
                "Online Delivery": row['online_order'],
                "Table booking": row['table booking'],
                "Address": row['local address'],
            },
            id = str(i)
        )

        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="zomato",
    persist_directory=db_path,
    embedding_function=embeddings
)

from tqdm import tqdm

if check_db:
    total = len(documents)
    MAX_BATCH = 5000  # safely below 5461 to avoid edge issues

    for i in tqdm(range(0, total, MAX_BATCH)):
        end = min(i + MAX_BATCH, total)
        batch_docs = documents[i:end]
        batch_ids = ids[i:end]

        vector_store.add_documents(documents=batch_docs, ids=batch_ids)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)

