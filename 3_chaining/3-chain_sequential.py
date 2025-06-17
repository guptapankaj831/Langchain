from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)

# Define Prompt template
animal_fact_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a facts expert who knows facts about {animal}.'),
        ('human', 'Tell me {fact_count} facts.')
    ]
)

# Define Prompt template for translation into french
translation_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a translator and convert the provided text into {language}.'),
        ('human', 'Translate the following text to {language} : {text}')
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"word count : {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {'text': output, 'language': 'french'})

# Create the combined chain using LangChain Expression Language (LCEL)
chain = animal_fact_template | chat_model | StrOutputParser() | prepare_for_translation | translation_template | chat_model | StrOutputParser()

result = chain.invoke({'animal': 'cat', 'fact_count': 2})
print(result)
