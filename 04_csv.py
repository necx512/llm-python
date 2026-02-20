from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Changed from OpenAI for tool-calling support
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.agents import create_csv_agent

load_dotenv()

# Change the dataset, original is not available for public
filepath = "academy/academy.csv"
loader = CSVLoader(filepath)
data = loader.load()
print(data)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # ChatOpenAI required for openai-tools agent

agent = create_csv_agent(llm, filepath, agent_type="openai-tools", verbose=True, allow_dangerous_code=True)  # Required in LangChain 1.x for code execution safety
agent.invoke({"input": "What percentage of the respondents are students versus professionals?"})
agent.invoke({"input": "List the top 3 devices that the respondents use to submit their responses"})
agent.invoke({"input": "Consider iOS and Android as mobile devices. What is the percentage of respondents that discovered us through social media submitting this from a mobile device?"})
