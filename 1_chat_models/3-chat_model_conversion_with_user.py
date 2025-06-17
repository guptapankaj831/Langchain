from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage('You are a helpful AI assistant')]    # Add system message
while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break

    chat_history.append(HumanMessage(content=query))                # Add Human message
    ai_response = chat_model.invoke(chat_history).content
    chat_history.append(AIMessage(content=ai_response))             # Add AI message
    print(f"AI: {ai_response}")
