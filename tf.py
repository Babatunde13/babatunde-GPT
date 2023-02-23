import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

class NeuralNetwork:
    def __init__(self, train_x, train_y) -> None:
        self.SAVED_MODEL_PATH = 'data/chatbot_model.h5'
        self.train_x = train_x
        self.train_y = train_y
        self.model = None
    
    def train(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.train_y[0]), activation='softmax'))
        self.sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
        self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=200, batch_size=5, verbose=1)
        self.model.save(self.SAVED_MODEL_PATH, self.model)
        print('model created')
    
    def get_model(self):
        if self.model is None:
            print("You  have not trained the model yet.")
            print("Loading the model from the saved file.")
            if os.path.exists(self.SAVED_MODEL_PATH):
                print("Model found. Loading the model.")
                self.model = load_model(self.SAVED_MODEL_PATH)
                print("Model loaded.")
            else:
                print("Model not found. Training the model and saving it.")
                self.train()
        return self.model
    
    def predict(self, sentence):
        self.model = self.get_model()
        self.bag = [0] * len(self.words)
        self.sentence_words = self.tokenizer.tokenize(sentence)
        self.sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in self.sentence_words]
        for word in self.sentence_words:
            for i, w in enumerate(self.words):
                if w == word:
                    self.bag[i] = 1
        self.results = self.model.predict(np.array([self.bag]))[0]
        self.results_index = np.argmax(self.results)
        self.tag = self.classes[self.results_index]
        return self.tag
