import tiktoken

with open("input.txt", "r") as file:
    text = file.read()

print("Input data length in characters:", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("Unique characters:", chars)
# print("Vocabulary size:", vocab_size)

# create a mapping from characters to indices and vice versa
# char_to_idx = {ch: i for i, ch in enumerate(chars)}
# idx_to_char = {i: ch for i, ch in enumerate(chars)}
# encode = lambda s: [char_to_idx[c] for c in s]
# decode = lambda l: ''.join([idx_to_char[i] for i in l])

tiktokenizer = tiktoken.get_encoding("gpt2")
print(tiktokenizer.encode("Hello, world!"))

# wrap into a torch tensor
import torch
data = torch.tensor(tiktokenizer.encode(text), dtype=torch.int64)
print(data.shape, data.dtype)

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.1, random_state=42)
print("Train data length in tokens:", len(train))
print("Test data length in tokens:", len(test))