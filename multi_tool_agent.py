from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.tools import Tool
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

class StreamCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end='', flush=True)  # Print streamed tokens in real-time

load_dotenv()

# 1. Define LLM
# llm = ChatOpenAI(model='gpt-3.5-turbo', callbacks=[StreamCallbackHandler()])
llm = ChatOpenAI(model='gpt-3.5-turbo')

# ------------ RAG Code Start ------------
def load_documents_from_folder(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(full_path)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(full_path)
        else:
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents

# Load and embed documents
folder_path = "docs"  # put .txt, .pdf, .docx files here
all_docs = load_documents_from_folder(folder_path)

spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = spliter.split_documents(all_docs)

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()

# Create the QA chain tool
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Conversational RAG Chain
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory
# )

# qa_tools = Tool(
#     name='MultiDocQA',
#     func=qa_chain.run,
#     description="Use this to answer questions based on uploaded documents (.pdf, .txt, .docx)"
# )

def debug_qa_tool(input_text):
    print("\n🛠️ QA Tool Invoked with input:", input_text)
    return qa_chain.invoke({"query": input_text})

qa_tools = Tool(
    name='MultiDocQA',
    func=debug_qa_tool,
    description="Use this to answer questions based on uploaded documents (.pdf, .txt, .docx) for Ansible and Ansible Playbook"
)

# ------------ RAG Code End ------------

# 2. Define Non-LLM Tool: calculator
@tool
def add_numbers(numbers: str) -> str:
    """Adds comma-separated numbers, e.g. '3, 5, 7'."""
    try:
        parts = [float(n.strip()) for n in numbers.split(",")]
        return str(sum(parts))
    except:
        return "Invalid input"

# Define Non-LLM Tool: explain how it works
@tool
def explain_addition(numbers: str) -> str:
    """Explains the step-by-step addition of numbers."""
    try:
        parts = [float(n.strip()) for n in numbers.split(",")]
        steps = " + ".join(str(n) for n in parts)
        return f"{steps} = {sum(parts)}"
    except:
        return "Invalid input"

# 3. Define LLM Tool: Summarize Text
@tool
def summarize_text(text: str) -> str:
    """Summarizes the given paragraph."""
    prompt = ChatPromptTemplate.from_template("Summarize this:\n\n{text}")
    return (prompt | llm).invoke({'text': text}).content

# Define LLM Tool: Suggest Title
@tool
def suggest_title(text: str) -> str:
    """Suggests a short, catchy title for the given paragraph."""
    prompt = ChatPromptTemplate.from_template("Suggest a short and catchy title for:\n\n{text}")
    return (prompt | llm).invoke({'text': text}).content

# Define LLM Tool: Summarize File Content
@tool
def summarize_file_content(file_path: str) -> str:
    """
    Summarizes the content of a text file using LLM. 
    Input should be a valid .txt file path.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    if not file_path.endswith(".txt"):
        return "Only .txt files are supported."

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = f"Summarize this content briefly:\n\n{content}"
    return llm.invoke(prompt)

# Create Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. List of tools/agent
tools = [add_numbers, explain_addition, summarize_text, suggest_title, summarize_file_content, qa_tools]

# 5. Custom Prompt (as required)
prompt = ChatPromptTemplate([
    ("system", "You're a helpful math assistant. Use tools only if the answer is not already available in the previous conversation. Use relevant product documentation if available. If a tool provides useful information, incorporate it into your final answer."),
    MessagesPlaceholder(variable_name='chat_history'),  # for past user/AI messages (from memory)
    ("user", "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad')   # for tool call traces and intermediate steps, necessary for agent reasoning
])

# 6. Create Agent using custom prompt
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 6. Run Agent
# user_question = "Can you add 12, 33, and 45 and explain how?"
user_question = "Here is a paragraph: LangChain is a powerful framework that helps developers build applications using LLMs easily. Can you summarize it?"
# user_question = "Suggest a good title for this: LangChain helps developers integrate LLMs into their applications with ease."
# user_question = "Add 5, 15, 25 and also summarize: LangChain is amazing for chaining LLMs with tools. Suggest a title too."
# user_question = "Hi there, just saying hello!"
user_question = "Write a short poem about math and compute 5*6."
user_question = "What is Ansible Playbook?"
response = agent_executor.invoke({'input': user_question})

# 7. Output
print("\n🤖 Agent Response:\n")
print(response)


# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
#     result = agent_executor.invoke({"input": user_input})
#     print("Agent:", result['output'])
#     print(memory.buffer)

