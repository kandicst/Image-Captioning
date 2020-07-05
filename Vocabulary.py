from collections import defaultdict
import re


class Vocabulary:

    def __init__(self, text, top_words=10000):
        self.word_freq = defaultdict(int)
        self.word_to_idx = {"<pad>": 0, "<start>": 1, "<end>": 2, '<unk>': 3}
        self.idx_to_word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: '<unk>'}
        self.encoded_captions = []
        self.unk_idx = self.word_to_idx['<unk>']
        self.top_words = top_words

        self.build_vocabulary(text)

    def build_vocabulary(self, text):
        self.build_word_frequencies(text)
        self.build_word_mappings()
        self.build_encoded_sentences(text)

    def build_encoded_sentences(self, text):
        self.encoded_captions = [self.encode_sentence(x) for x in text]

    def build_word_frequencies(self, text):
        for sentence in text:
            sentence = self.preprocess_sentence(sentence)
            for word in sentence:
                if word in self.word_freq:
                    self.word_freq[word] += 1
                else:
                    self.word_freq[word] = 1

        self.word_freq.pop("<start>", None)
        self.word_freq.pop("<end>", None)

    def preprocess_sentence(self, caption):
        caption = caption.lower()
        caption = re.sub('[^A-Za-z0-9 <>]+', '', caption)
        return caption.split()

    def build_word_mappings(self):
        start_idx = len(self.word_to_idx)
        for word in sorted(self.word_freq, key=self.word_freq.get, reverse=True):
            self.word_to_idx[word] = start_idx
            self.idx_to_word[start_idx] = word
            start_idx += 1
            if start_idx >= self.top_words: break

    def encode_sentence(self, sentence):
        return [self.word_to_idx.get(x, self.unk_idx) for x in self.preprocess_sentence(sentence)]

    def decode_sentence(self, indices):
        return " ".join([self.idx_to_word.get(x, '<unk>') for x in indices])


# if __name__ == '__main__':
    # preprocess_sentence('<START>........... A very clean and well decorated empty bathroom <END>')
