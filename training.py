import random, json, pickle, os, nltk, numpy as np
from lemmatizer import CustomLemmatizer
from tf import NeuralNetwork
from tokenizer import CustomTokenizer

class TokenizeAndLemmatizeIntents:
    def __init__(self, intents):
        # punkt is a pre-trained model that is used to tokenize words
        PRE_TRAINED_MODEL_DIR = './data'
        self.WORDS_SERIALIZATION_PATH = os.path.join(PRE_TRAINED_MODEL_DIR, 'words.pickle')
        self.CLASSES_SERIALIZATION_PATH = os.path.join(PRE_TRAINED_MODEL_DIR, 'classes.pickle')
        self.DOCUMENTS_SERIALIZATION_PATH = os.path.join(PRE_TRAINED_MODEL_DIR, 'documents.pickle')
        if not os.path.exists(PRE_TRAINED_MODEL_DIR):
            os.makedirs(PRE_TRAINED_MODEL_DIR)

        nltk.data.path.append(PRE_TRAINED_MODEL_DIR)
        nltk.download('punkt', download_dir=PRE_TRAINED_MODEL_DIR)
        # wordnet is a lexical database for the English language
        nltk.download('wordnet', download_dir=PRE_TRAINED_MODEL_DIR)

        self.intents = intents
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',', ':']
        self.lemmatizer = CustomLemmatizer()
        self.tokenizer = CustomTokenizer()

    def tokenize_and_lemmatize(self):
        for intent in self.intents['intents']:
            self.classes.append(intent['tag'])
            for pattern in intent['patterns']:
                word_list = self.tokenizer.tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))

        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(set(self.words))

        self.classes = sorted(set(self.classes))

    def save_words(self):
        with open(self.WORDS_SERIALIZATION_PATH, 'wb') as f:
            pickle.dump(self.words, f)
    
    def save_classes(self):
        with open(self.CLASSES_SERIALIZATION_PATH, 'wb') as f:
            pickle.dump(self.classes, f)
    
    def save_documents(self):
        with open(self.DOCUMENTS_SERIALIZATION_PATH, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def save(self):
        self.save_words()
        self.save_classes()
        self.save_documents()
    
    def get_words(self):
        return self.words
    
    def get_classes(self):
        return self.classes
    
    def get_documents(self):
        return self.documents

class WordTransformer:
    def __init__(self) -> None:
        intents = json.loads(open('intents.json').read())
        tokenize_and_lemmatize_intents = TokenizeAndLemmatizeIntents(intents)

        tokenize_and_lemmatize_intents.tokenize_and_lemmatize()
        tokenize_and_lemmatize_intents.save()

        self.lemmatizer = CustomLemmatizer()
        self.documents = tokenize_and_lemmatize_intents.get_documents()
        self.classes = tokenize_and_lemmatize_intents.get_classes()
        self.words = tokenize_and_lemmatize_intents.get_words()
        self.training = []
        self.output_empty = [0] * len(self.classes)

    def transform(self):
        for doc in self.documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            output_row = list(self.output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.training.append([bag, output_row])

        random.shuffle(self.training)
        self.training = np.array(self.training)
        train_x = list(self.training[:, 0])
        train_y = list(self.training[:, 1])
        return train_x, train_y

wordTransformer = WordTransformer()
train_x, train_y = wordTransformer.transform()
neuralNetwork = NeuralNetwork(train_x, train_y)
neuralNetwork.train()
