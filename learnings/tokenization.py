from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Zomato delivers food to your doorstep."
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(tokens)
print("Tokens:", tokens)
print("IDs:", ids)