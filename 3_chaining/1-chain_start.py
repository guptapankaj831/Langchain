# 1. Extended or Sequential Chaining
# 2. Parallel Chaining
# 3. Conditional/Branching Chaining

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)

# Define Prompt template (Remember: No need for separate Runnable Chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a facts expert who knows facts about {animal}.'),
        ('human', 'Tell me {fact_count} facts.')
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | chat_model | StrOutputParser()

# StrOutputParser() - this function will extract value of 'content' property from the response of previous task.

# Run the chain
result = chain.invoke({"animal": "cat", "fact_count": 3})

# No need to use 'result.content', as we are using StrOutputParser()
print(result)

