import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions.categorical import Categorical


class Baseline(nn.Module):
    """
    Baseline model for doing image captioning using an CNN as encoder and RNN as decoder.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, vanilla=False, embeddings=None):
        super(Baseline, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, vocab_size, vanilla, embeddings)

    def forward(self, image, captions, length):
        feature_vector = self.encoder(image)
        tag_scores = self.decoder(feature_vector, captions, length)
        return tag_scores

    def sample(self, images, sampling=False, temperature=1.0, max_sentence_length=18):
        feature_vectors = self.encoder(images)
        return self.decoder.sample(feature_vectors, sampling, temperature, max_sentence_length)


class Encoder(nn.Module):
    """
    Use transfer learning with ResNet50 to encode the images to a feature vector of size of the embeddings.
    """

    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        # Load a pre-trained ResNet50 and freeze weights
        self.encoder = models.resnet50(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Substitute trainable linear layer
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)

    def forward(self, image):
        return self.bn(self.encoder(image))


class Decoder(nn.Module):
    """
    Use an RNN as decoder for doing the tagging of the images.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, vanilla=False, embeddings=None):
        super(Decoder, self).__init__()
        if embeddings is None:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        # The RNN takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        if not vanilla:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence, captions, length):
        embeds = self.word_embeddings(captions)
        embeds = torch.cat((sentence.unsqueeze(1), embeds), 1)
        packed = pack_padded_sequence(embeds, length, batch_first=True)
        hiddens, _ = self.rnn(packed)
        tag_space = self.hidden2tag(hiddens[0])
        return tag_space

    def sample(self, features, sampling, temperature, max_sentence_length):
        features = features.unsqueeze(1)
        vocab_indexes = []
        # generate word by word
        states = None
        for i in range(max_sentence_length):
            hiddens, states = self.rnn(features, states)
            outputs = self.hidden2tag(hiddens.squeeze(1))
            if sampling:
                # Stochastic
                outputs = outputs / temperature
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = Categorical(probs).sample()
            else:
                # Deterministic
                _, preds = outputs.max(1)

            vocab_indexes.append(preds)
            features = self.word_embeddings(preds)
            features = features.unsqueeze(1)
        vocab_indexes = torch.stack(vocab_indexes, 1)
        return vocab_indexes
