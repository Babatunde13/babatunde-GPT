from chatbot import ChatBot

def main():
    chatbot = ChatBot()
    print('Chatbot is ready to talk! (type quit to stop)')
    message = ''
    while message != 'quit':
        message = input('You: ')
        print('Chatbot: ' + chatbot.chat(message))

if __name__ == '__main__':
    main()
