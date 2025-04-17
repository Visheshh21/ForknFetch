import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
import torch 
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device set to ', device)
df=pd.read_csv('data.csv',engine='python', on_bad_lines='warn')
df['text'] = df['Name'] + '. ' + df['Description'].fillna('') + '. Ingredients: ' + df['RecipeIngredientParts'].fillna('') + '. Instructions: ' + df['RecipeInstructions'].fillna('')
documents = [
    Document(page_content=row['text'], metadata={"name": row['Name'], "calories": row.get('Calories', 'N/A')})
    for _, row in df.iterrows()
]
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device})

db = FAISS.from_documents(documents, embedding=embedder)
db.save_local("faiss_index/chef_recipes")
print("FAISS index saved to faiss_index/chef_recipes")