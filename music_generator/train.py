# PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam

# Custom
from utils import SlidingWindowLoader, to_onehot
from generator import sample

# Other
import random


def fit(model, train_encoded, val_encoded, config):
    """
    Fit the models weights and save the training and validation loss in the model
    :param model: nn. Module
    :param train_encoded: Encoded training data
    :param val_encoded: Encoded validation data
    :param config: dict with settings
    :return:
    """
    n_songs_train = len(train_encoded)
    n_songs_val = len(val_encoded)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

    for epoch in range(1, config["EPOCHS"] + 1):
        train_loss = 0

        # Enter train mode to activate Dropout and Batch Normalization layers
        model.train()

        # Shuffle songs for each epoch
        random.shuffle(train_encoded)
        for i, song in enumerate(train_encoded):
            # Reset state for each song
            model.init_state()

            song_loss = 0
            n = 0  # Number of chunks made from song
            for seq, target in SlidingWindowLoader(song, window=config["CHUNK_SIZE"]):

                # Chunks is sometimes empty
                if len(seq) == 0:
                    continue
                n += 1

                # One-hot encode chunk tensor
                input_onehot = to_onehot(seq, config["VOCAB_SIZE"])

                optimizer.zero_grad()  # Reset gradient for every forward
                output = model(input_onehot.unsqueeze(1))  # Size = (chunk_length, batch, vocab_size)
                output.squeeze_(1)  # Back to 2D
                chunk_loss = criterion(output, target.long())
                chunk_loss.backward()
                optimizer.step()
                song_loss += chunk_loss.item()
            train_loss += song_loss / n
            if i % 100 == 0:
                print("Song: {}, AvgTrainLoss: {}".format(i, train_loss / (i + 1)))

        # Append average training loss for this epoch
        model.training_losses.append(train_loss / n_songs_train)

        # Generate a song at this epoch
        song = sample(model, "$", config)
        print("{}\n{}\n{}".format("-" * 40, song, "-" * 40))

        # Validation
        with torch.no_grad():
            print("Validating")
            model.eval()  # Turns of Dropout and BatchNormalization
            val_loss = 0

            for song in val_encoded:
                # Reset state
                model.init_state()

                song_loss = 0
                n = 0
                for seq, target in SlidingWindowLoader(song, window=config["CHUNK_SIZE"]):
                    # Chunks is sometimes empty
                    if len(seq) == 0:
                        continue
                    n += 1

                    # One-hot encode chunk tensor
                    input_onehot = to_onehot(seq, config["VOCAB_SIZE"])

                    output = model(input_onehot.unsqueeze(1))  # Size = (chunk_length, batch, vocab_size)
                    output.squeeze_(1)  # Back to 2D
                    song_loss += criterion(output, target.long()).item()
                val_loss += song_loss / n
            model.validation_losses.append(val_loss / n_songs_val)
            print("Epoch {}, Training loss: {}, Validation Loss: {}".format(epoch, model.training_losses[-1],
                                                                            model.validation_losses[-1]))
