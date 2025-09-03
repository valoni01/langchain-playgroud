from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    summary_template = """
    Given the information {information} about a person, I want you to create 
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": "John is a software engineer from California. He loves hiking and photography. His favorite food is sushi."})

    print(response.content)


if __name__ == "__main__":
    main()
