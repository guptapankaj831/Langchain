from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    # task="text-generation",
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)
print(chat_model.invoke('What is capital of india.?').content)
