from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)

# ------------------------ When we have only one type of template/message-------------------------------------
# 1. Write template
template = 'Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a strength. ' \
'Keep it to 4 lines max.'

# 2. Convert template into langchain understable format
prompt_template = ChatPromptTemplate.from_template(template)

# 3. Invoke prompt template to replace dynamic value
prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "samsung",
    "position": "AI Engineer",
    "skill": "AI"
})

#print(chat_model.invoke(prompt).content)
# ------------------------------------------------------------------------------------------------------------

# ------------------------ When we have multiple template/messages-------------------------------------
# 1. Write template/message list
messages = [
    ('system', 'You are a comedian who tells joke about {topic}'),
    ('human', 'Tell me {joke_count} jokes.')
]

# 2. Convert template into langchain understable format
prompt_template = ChatPromptTemplate.from_messages(messages)

# 3. Invoke prompt template to replace dynamic value
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

print(chat_model.invoke(prompt).content)
# ------------------------------------------------------------------------------------------------------------

