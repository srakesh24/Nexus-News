from sentence_transformers import SentenceTransformer 
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["Zomato delivers food to your doorstep.", "Python is a programming language.",]

embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")

print(f"First Embedding (First 5 values): {embeddings[0][:5]}...")

