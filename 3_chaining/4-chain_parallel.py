from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4o-mini')

# Define Prompt template for movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a movie critic.'),
        ('human', 'Provide a brief summary of the movie {movie_name}.')
    ]
)

# Define plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a movie critic.'),
            ('human', 'Analyze the plot: {plot}. What are the strengths and weaknesses')
        ]
    )
    return plot_template.format_prompt(plot=plot)

# Define Character analysis step
def analyze_character(character):
    character_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a movie critic.'),
            ('human', 'Analyze the Characters: {character}. What are the strengths and weaknesses')
        ]
    )
    return character_template.format_prompt(character=character)

# Combine analyses into a final verdict
def combine_verdict(plot_analyze, character_analyze):
    return f"Plot Analysis: \n {plot_analyze} \n\n\n Character Analysis: \n{character_analyze}"

# Simplify branches with LCEL
plot_branch_chain = (RunnableLambda(lambda x: analyze_plot(x)) | chat_model | StrOutputParser())
character_branch_chain = (RunnableLambda(lambda x: analyze_character(x)) | chat_model | StrOutputParser())

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    summary_template
    | chat_model
    | StrOutputParser()
    | RunnableParallel(branches={'plot': plot_branch_chain, 'character': character_branch_chain})
    | RunnableLambda(lambda x: combine_verdict(x['branches']['plot'], x['branches']['character']))
)

result = chain.invoke({'movie_name': '3 idiots'})
print(result)
