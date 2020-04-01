import numpy as np
from build_vocab import Vocabulary
import torch.nn as nn
import torch

# Read weights per word to a dictionary
file = open("./glove.6B/glove.6B.50d.txt", "r")
word2weight = dict()
for line in file:
    line = line.split()
    word = line[0]
    weights = list(map(float, line[1:]))
    word2weight[word] = weights
file.close()

# Load target vocabulary
train_caption_path = "./data/annotations/captions_train2014.json"
vocab_path = "./vocabulary.pkl"
vocab = Vocabulary(train_caption_path, vocab_path)

words_in_vocab = vocab.word_to_id.keys()

# Find matching words between vocab and pre-trained and save their weights. Else random weights
embedding_dim = 50  # Must fit the dimension from the pre-trained file
pretrained_weight_matrix = np.zeros((len(words_in_vocab), embedding_dim))
words_found = 0
for i, word in enumerate(words_in_vocab):
    try:
        pretrained_weight_matrix[i] = word2weight[word]
        words_found += 1
    except KeyError:
        pretrained_weight_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

print("{} out of {} words matched".format(words_found, len(vocab)))

# Convert to tensor
pretrained_weight_matrix = torch.from_numpy(pretrained_weight_matrix)

# How to use these weight as embedding
emb_layer = nn.Embedding.from_pretrained(pretrained_weight_matrix, freeze=True)
