from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

input_data = {
    "resume_summary": "Experienced software engineer with a strong background in Python, AI, and backend development.",
    "job_title": "AI Developer",
    "company_name": "OpenAI",
    "job_description": (
        "Looking for an AI Developer to build intelligent apps using LLMs, Python, and LangChain. "
        "Should have experience with prompt engineering, API integration, and scalable app development."
    )
}

cover_letter_prompt = PromptTemplate.from_template("""
Write a formal and enthusiastic cover letter for the following job application:

Candidate Summary:
{resume_summary}

Target Job Title:
{job_title}

Company:
{company_name}

Job Description:
{job_description}

The cover letter should highlight relevant experience, express excitement for the role, and be concise and well-formatted.
""")

llm = ChatOpenAI(model='gpt-3.5-turbo')

cover_letter_chain = cover_letter_prompt | llm | StrOutputParser()
response = cover_letter_chain.invoke(input_data)

print(response)
