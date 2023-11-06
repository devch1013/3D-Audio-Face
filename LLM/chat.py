import configparser
import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


ai_secret_file = "/home/ubuntu/3d_temp/LLM/keys.ini"
config = configparser.ConfigParser()
config.read(ai_secret_file)
openai_api_key = config['OPENAI']['OPENAI_API_KEY']
os.environ.update({'OPENAI_API_KEY': openai_api_key})


class ChatGPTConversation:
    def __init__(self):
        llm = OpenAI(temperature=0)
        self.conversation = ConversationChain(
            llm=llm, verbose=True, memory=ConversationBufferMemory()
        )
        
    def __call__(self, prompt):
        text = self.conversation.predict(input=prompt)
        return text
    
    
if __name__ == "__main__":
    conversation_agent = ChatGPTConversation()
    print(conversation_agent("hello my name is chanhyuk"))
    print(conversation_agent("what is my name??"))