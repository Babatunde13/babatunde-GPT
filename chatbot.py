import random, json, pickle, numpy as np
from lemmatizer import CustomLemmatizer
from tokenizer import CustomTokenizer
from tf import load_model

class ChatBot:
    def __init__(self) -> None:
        self.intents = json.loads(open('intents.json').read())

        self.words = pickle.load(open('data/words.pickle', 'rb'))
        self.classes = pickle.load(open('data/classes.pickle', 'rb'))
        self.documents = pickle.load(open('data/documents.pickle', 'rb'))

        self.model = load_model('data/chatbot_model.h5')

    def clean_up_sentence(self, sentence):
        lemmatizer = CustomLemmatizer()
        tokenizer = CustomTokenizer()
        sentence_words = tokenizer.tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % word)
        return(np.array(bag))

    def predict_class(self, sentence, model):
        p = self.bag_of_words(sentence, self.words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result

    def chat(self, message):
        ints = self.predict_class(message, self.model)
        res = self.get_response(ints, self.intents)
        return res
    
    def get_intents(self):
        return self.intents

