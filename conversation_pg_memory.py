from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Set up PostgreSQL connection
pg_history = PostgresChatMessageHistory(
    connection_string=os.environ.get("POSTGRES"),
    session_id="user_123",
    table_name="langchain_chat_messages"
)

# Attach to ConversationBufferMemory
pg_memory = ConversationBufferMemory(
    chat_memory=pg_history,
    return_messages=True
)

# Define the LLM
llm = ChatOpenAI(temperature=0.7)

# Create the ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory=pg_memory,
    verbose=True
)

# Run some interactions
#print(conversation.run("Hello, I’m Riya."))
#print(conversation.run("I’m interested in product design."))
#print(conversation.run("Can you remind me what I told you earlier?"))
print(conversation.run("What you remember about Riya.?"))
