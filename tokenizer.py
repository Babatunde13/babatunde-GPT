import nltk

class CustomTokenizer:
    def __init__(self):
        self.tokenizer = nltk.word_tokenize

    def tokenize(self, word):
        return self.tokenizer(word)
