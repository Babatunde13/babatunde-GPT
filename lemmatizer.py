from nltk.stem import WordNetLemmatizer

class CustomLemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)
