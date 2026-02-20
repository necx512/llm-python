from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Changed from OpenAI for tool-calling support
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()

# dburi = os.getenv("DATABASE_URL")
# Change the dataset, original is not available for public
dburi = "sqlite:///academy/academy.db"
db = SQLDatabase.from_uri(dburi)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # ChatOpenAI required for openai-tools agent

# SQLDatabaseChain is deprecated in LangChain 1.x, use create_sql_agent instead
db_chain = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

db_chain.invoke({"input": "How many rows is in the responses table of this db?"})
db_chain.invoke({"input": "Describe the responses table"})
db_chain.invoke({"input": "What are the top 3 countries where these responses are from?"})
db_chain.invoke({"input": "Give me a summary of how these customers come to hear about us. \
    What is the most common way they hear about us?"})
