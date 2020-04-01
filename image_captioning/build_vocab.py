from pycocotools.coco import COCO
import nltk
import pickle
import os.path

class Vocabulary:
    # Usage: v = Vocabulary('./data/annotations/captions_val2014.json')
    def __init__(self, captions_path, vocab_path):
        self.word_to_id, self.id_to_word = dict(), list()
        self.vocab_path = vocab_path

        if not os.path.isfile(vocab_path):
            self.coco = COCO(captions_path)
            self.make_table()
        else:
            self.load_vocab(vocab_path)

    def load_vocab(self, path):
        with open(path, 'rb') as f:
            self.word_to_id, self.id_to_word = pickle.load(f)

    def add_word(self, word, words):
        if word not in words:
            words.add(word)
            self.id_to_word.append(word)
            self.word_to_id[word] = len(self.id_to_word) - 1
        
    def make_table(self):
        """
        Make a table of all captions mapping to an encoding
        :param captions:
        :return:
        """
        anns = self.coco.anns
        ids = list(anns.keys())
        
        words = set()
        # Padding is indexed at 0
        self.add_word("<padding>", words)
        # Unknown word is indexed at 1
        self.add_word("<unknown>", words)
        self.add_word("<start>", words)
        self.add_word("<end>", words)

        # Add the result of words
        for i in ids:
            split_words = nltk.word_tokenize(str(anns[i]['caption']).lower())
            for word in split_words:
                if word not in words:
                    self.add_word(word, words)

        with open(self.vocab_path, 'wb') as f:
            pickle.dump((self.word_to_id, self.id_to_word), f)

    def vocab(self, token):
        """

        :param token: A word which is going to b
        :return:
        """
        return self.word_to_id[token] if token in self.word_to_id else self.word_to_id["<unknown>"]
    
    def index(self, ind):
        return self.id_to_word[ind]

    def get_sentence(self, indexes):
        words = []
        for i in indexes:
            word = self.index(i)
            if word == "<start>":
                continue
            if word == "<end>":
                break
            words.append(word)
        return " ".join(words)
#         return " ".join([self.index(i) for i in indexes])

    def __len__(self):
        return len(self.id_to_word)