from talk_module.LLM.chat import ChatGPTConversation


if __name__ == "__main__":
    conversation_agent = ChatGPTConversation(verbose=False)
    print("""
Choose the emotion of AI chatbot
1: "extremely happy and friendly. love user",
2: "happy",
3: "neutral",
4: "annoying and not friendly",
5: "extremely angry, annoying and absolutley not friendly. hate user",
          """)
    emotion = input("Enter the number of emotion: ")
    emotion_dict = {1: "extremely happy and friendly. love user",
    2: "happy",
    3: "neutral",
    4: "annoying and not friendly",
    5: "extremely angry, annoying and absolutley not friendly. hate user"}
    conversation_agent.change_emotion(emotion_dict[int(emotion)])
    print("Start conversation with AI chatbot")
    while True:
        user_input = input("user: ")
        print("AI: ",conversation_agent(user_input))
