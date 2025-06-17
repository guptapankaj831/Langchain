from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage('You are expert in social media content strategy'),
    HumanMessage('Give a short tip to create engaging posts on Instagram')
]

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)
print(chat_model.invoke(messages).content)
