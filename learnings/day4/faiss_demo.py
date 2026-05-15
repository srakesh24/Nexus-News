import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "How to reset your password",
    "Track your Zomato delivery live",
    "Cancel a Zomato order before pickup",
    "Zomato Gold membership benefits",
    "Contact Zomato customer support"
]

# Convert documents into embeddings
doc_embeddings = model.encode(docs).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])

# Store embeddings
index.add(doc_embeddings)

# User query
query = "I want to cancel my food order"

# Query embedding
q_emb = model.encode([query]).astype('float32')

# Similarity search
distances, indices = index.search(q_emb, k=2)


# Show results
for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}: {docs[idx]} (score: {distances[0][i]:.2f})")