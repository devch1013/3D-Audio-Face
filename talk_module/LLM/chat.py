import configparser
import os

# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate


ai_secret_file = "talk_module/LLM/keys.ini"
config = configparser.ConfigParser()
config.read(ai_secret_file)
openai_api_key = config["OPENAI"]["OPENAI_API_KEY"]
os.environ.update({"OPENAI_API_KEY": openai_api_key})


class ChatGPTConversation:
    def __init__(self, template_file="emotion_template.txt"):
        llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        template_file = f"talk_module/LLM/templates/{template_file}"
        with open(template_file, "r") as f:
            template = f.read()
        prompt_template = PromptTemplate(
            input_variables=["emotion", "input", "history"], template=template
        )
        self.emotion = "neutral"
        self.conversation = LLMChain(
            prompt=prompt_template,
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory(memory_key="history", input_key="input"),
        )

    def __call__(self, prompt):
        # self.conversation.memory.clear()
        text = self.conversation.predict(input=prompt, emotion=self.emotion)
        return text
    
    def change_emotion(self, prompt):
        self.emotion = prompt
        
    def reset_history(self):
        self.conversation.memory.clear()


if __name__ == "__main__":
    conversation_agent = ChatGPTConversation()
    print(conversation_agent("hello my name is chanhyuk"))
    print(conversation_agent("what is my name??"))
