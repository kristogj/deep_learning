import torch
import numpy as np
import logging
from functools import reduce
import time
from torch.nn.utils.rnn import pack_padded_sequence

# Custom
from evaluate_captions import evaluate_captions
from plotting import generate_lineplot
from tqdm import tqdm, tqdm_notebook
import json


def train(model, optimizer, criterion, train_loader, val_loader, vocab, config):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    for epoch in range(config["epochs"]):
        ts = time.time()
        train_loss = 0
        num_batches = 0
        for iter, (images, captions, lengths) in enumerate(train_loader):
            # This will return (images, captions, lengths) for each iteration.
            # images: a tensor of shape (batch_size, 3, 224, 224).
            # captions: a tensor of shape (batch_size, padded_length).
            # lengths: a list indicating valid length for each caption. length is (batch_size).

            # Set data to device
            images = images.to(config["device"])
            captions = captions.to(config["device"])
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # 1. Clear accumulated gradients
            model.zero_grad()

            # 2. Predict the tag_scores from the images
            tag_scores = model(images, captions, lengths)

            # 3. Compute the loss, gradients, and update parameters
            loss = criterion(tag_scores, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if iter % 10 == 0:
                logging.info("Epoch{}, iter{}, loss: {}".format(epoch, iter, train_loss / num_batches))

        train_loss /= num_batches
        logging.info("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        # Validate for each epoch
        val_loss = val(model, criterion, val_loader, epoch, vocab, config)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'best_model.pth')

        # Append losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check if errors in val_errors are strictly increasing
        LARGE_NUM = np.inf
        if len(val_losses) > 4 and reduce(lambda i, j: j if i < j else LARGE_NUM, val_losses[-5:]) != LARGE_NUM:
            logging.info("Early stopped at epoch {}".format(epoch))
            break

    logging.info("Generating plot to baseline_train_val_loss.png")
    generate_lineplot((train_losses, val_losses), "Loss", "baseline_train_val_loss.png")

    return None


def val(model, criterion, val_loader, epoch, vocab, config):
    logging.info("=====================================")
    logging.info("Validating at epoch {}".format(epoch))
    model.eval()
    with torch.no_grad():
        val_loss = 0
        num_batches = 0
        for iter, (images, captions, lengths) in enumerate(val_loader):
            # Set data to device
            images = images.to(config["device"])
            captions = captions.to(config["device"])
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Do one sample to see progress on taggings
            if iter == 0:
                vocab_indexes = model.sample(images, config["sampling"], config["temperature"],
                                         config["max_sentence_length"])
                logging.info("Captions generated {}".format(
                    [vocab.get_sentence(vocab_indexes[i, :]) for i in range(len(vocab_indexes))]))

            # Predict
            tag_scores = model(images, captions, lengths)
            loss = criterion(tag_scores, targets)
            val_loss += loss.item()
            num_batches += 1
        val_loss /= num_batches
    model.train()
    logging.info("Avg loss: {}".format(val_loss))
    logging.info("=====================================")
    return val_loss


def test(model, criterion, test_loader, vocab, config):
    model.eval()
    # See what the scores are after training
    with torch.no_grad():
        test_loss = 0
        num_batches = 0
        captions = {}
        for image, caption, img_id in tqdm(test_loader):
            length = [caption.shape[1]]
            # Set data to device
            image = image.to(config["device"])
            caption = caption.long().to(config["device"])
            targets = pack_padded_sequence(caption, length, batch_first=True)[0]

            tag_scores = model(image, caption, length)
            loss = criterion(tag_scores, targets)
            test_loss += loss.item()
            num_batches += 1

            # Save each caption to file
            vocab_indexes = model.sample(image, config["sampling"], config["temperature"],
                                         config["max_sentence_length"])
            captions[img_id.item()] = vocab.get_sentence(vocab_indexes[0, :])

        test_loss /= num_batches
    model.train()

    pred_annotations_file = 'baseline_lstm_captions.json'
    with open(pred_annotations_file, 'w') as f:
        json.dump(captions, f)

    perplexity = np.exp(test_loss)
    logging.info("Test loss: {}".format(test_loss))
    logging.info("Perplexity score: {}".format(perplexity))

    true_annotations_file = './data/annotations/captions_val2014.json'
    BLEU1, BLEU4 = evaluate_captions(true_annotations_file, pred_annotations_file)
    logging.info("BLEU 1: {}".format(np.round(BLEU1, 2)))
    logging.info("BLEU 4: {}".format(np.round(BLEU4, 2)))

    return BLEU1, BLEU4
