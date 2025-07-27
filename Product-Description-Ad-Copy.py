from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. Setup LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 2. Prompt for Description
desc_prompt = PromptTemplate.from_template(
    "Write a detailed product description for: {product}"
)

# 3. Prompt for Ad Copy
ad_prompt = PromptTemplate.from_template(
    "Write a catchy social media ad for this product description:\n\n{description}"
)

# 4. Chains
desc_chain = desc_prompt | llm
ad_chain = ad_prompt | llm

# 5. Final chain: product ➝ description ➝ ad copy
full_chain = (
    RunnableLambda(lambda x: x)                             # just pass product key
    | desc_chain                                            # gets description
    | RunnableLambda(lambda x: {'description': x.content})  # pass only text as {description}
    | ad_chain                                              # gets ad copy
)

# full_chain = (
#     RunnableLambda(lambda x: x)                             # just pass product key
#     | desc_prompt | llm                                            # gets description
#     | RunnableLambda(lambda x: {'description': x.content})  # pass only text as {description}
#     | ad_prompt | llm                                              # gets ad copy
# )

# 6. Input
input_data = {"product": "Smart Water Bottle with Hydration Reminder"}

# 7. Run it
result = full_chain.invoke(input_data)

# 8. Output
print("\n📢 Final Ad Copy:\n")
print(result)
