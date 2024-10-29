# app.py

import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Load your dataset
data = pd.read_csv('Hydra-Movie-Scrape.csv')

# Prepare the text data for indexing
text_data = data['Summary'].fillna('') + " " + data['Short Summary'].fillna('')

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(text_data.tolist(), convert_to_tensor=True).cpu().detach().numpy()

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Groq API Key and Endpoint
GROQ_API_KEY = "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" 
GROQ_API_ENDPOINT = "https://api.groq.com/v1/meta-llama-3-8b-instruct/completions"  # Update endpoint as needed

# Define the RAG function
def rag_query(query, top_k=5):
    # Step 1: Get query embedding
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().detach().numpy()

    # Step 2: Search for the most relevant entries in the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]]

    # Step 3: Generate a context string from retrieved documents
    context = " ".join(results['Summary'][:top_k].fillna(''))

    # Step 4: Send the query to Groq API with the context
    payload = {
        "prompt": f"Question: {query}\nContext: {context}",
        "max_tokens": 50
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(GROQ_API_ENDPOINT, json=payload, headers=headers)
    response_data = response.json()

    # Extract and return the response text
    response_text = response_data["choices"][0]["text"]
    return response_text

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# Good for Movie lovers")
    gr.Markdown("Using Hydra-Movie-Scrape dataset, this RAG app will answer questions about movies, directors, actors, etc.")
    
    query = gr.Textbox(label="Enter your query here:")
    response = gr.Textbox(label="Response:")

    query_button = gr.Button("Get Answer")

    def get_answer(query):
        return rag_query(query)

    query_button.click(fn=get_answer, inputs=query, outputs=response)

# Launch the Gradio interface
interface.launch()
