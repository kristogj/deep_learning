import math

import torch
from torch.utils.data.dataset import Dataset


class SlidingWindowLoader(Dataset):
    def __init__(self, data, window=100):
        self.data = data
        self.window = window
        self.current = 0
        self.high = self.__len__()

    def __getitem__(self, index):
        index_pos = index * self.window
        x = self.data[index_pos:min(len(self.data) - 1, index_pos + self.window)]
        target = self.data[index_pos + 1:index_pos + 1 + self.window]
        return x, target

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if (self.current - 1) < self.high:
            return self.__getitem__(self.current - 1)
        raise StopIteration

    def __len__(self):
        return math.ceil(len(self.data) / self.window)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def encode_songs(songs, char_to_idx):
    """
    Return a list of encoded songs where each char in a song is mapped to an index as in char_to_idx
    :param songs: List[String]
    :param char_to_idx: Dict{char -> int}
    :return: List[Tensor]
    """
    songs_encoded = [0] * len(songs)
    for i, song in enumerate(songs):
        chars = list(song)
        result = torch.zeros(len(chars)).to(get_device())
        for j, ch in enumerate(chars):
            result[j] = char_to_idx[ch]
        songs_encoded[i] = result
    return songs_encoded


def to_onehot(t, vocab_size):
    """
    Take a list of indexes and return a one-hot encoded tensor
    :param vocab_size: Size of one hot encoding
    :param t: 1D Tensor of indexes
    :return: 2D Tensor
    """
    inputs_onehot = torch.zeros(t.shape[0], vocab_size).to(get_device())
    inputs_onehot.scatter_(1, t.unsqueeze(1).long(), 1.0)  # Remember inputs is indexes, so must be integer
    return inputs_onehot


def char_mapping():
    """
    Mapping each unique char to an index for one-hot encoding
    :return: Dict{char -> index}, Dict{index -> char}
    """
    file = open("data/train.txt")
    text = file.read()
    text = text.replace("<start>", "$")
    text = text.replace("<end>", "%")
    chars = list(set(text))
    chars.sort()  # To get the same order every time
    file.close()

    vocab_size = len(chars)
    print("Data has {} unique characters".format(vocab_size))

    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    return char_to_ix, ix_to_char


def read_songs_from(file_name):
    with open(file_name, 'r') as songs_file:
        songs = songs_file.read()
        songs = songs.replace("<start>", "$")
        songs = songs.replace("<end>", "%")
    song_delimiter = '%'
    songs = songs.split(song_delimiter)[:-1]
    songs = [song + song_delimiter for song in songs]
    return songs


def negative_log_likelihood(model, encoded_data, criterion, config):
    """
    Average the cross entropy loss over all the chunks
    :param model: nn.Module
    :param encoded_data: List of encoded songs
    :return:
    """
    chunk_loss = 0
    number_of_chunks = 0
    with torch.no_grad():
        model.eval()
        for song in encoded_data:
            model.init_state()
            for seq, target in SlidingWindowLoader(song, window=config["CHUNK_SIZE"]):
                number_of_chunks += 1
                if len(seq) == 0:
                    continue
                inputs_onehot = to_onehot(seq, config["VOCAB_SIZE"])
                output = model(inputs_onehot.unsqueeze(1))  # Turn input into 3D (chunk_length, batch, vocab_size)
                output.squeeze_(1)  # Back to 2D
                chunk_loss += criterion(output, target.long())
    return chunk_loss / number_of_chunks
