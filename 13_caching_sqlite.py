import time
from dotenv import load_dotenv
import langchain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.cache import SQLiteCache
from langchain_classic.chains.summarize import load_summarize_chain  # Moved to langchain_classic in LangChain 1.x

# add this to .gitignore if you don't want to commit the cache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

load_dotenv()

text_splitter = CharacterTextSplitter()
llm = OpenAI(model="gpt-3.5-turbo-instruct")  # Updated from deprecated text-davinci-002
no_cache_llm = OpenAI(model="gpt-3.5-turbo-instruct", cache=False)

with open("news/summary.txt") as f:
    news = f.read()

texts = text_splitter.split_text(news)
print(texts)

docs = [Document(page_content=t) for t in texts[:3]]

chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)

with get_openai_callback() as cb:
    start = time.time()
    result = chain.invoke({"input_documents": docs})  # LangChain 1.x uses .invoke()
    end = time.time()
    print("--- result1")
    print(result["output_text"])  # Extract text from result dict
    print(str(cb) + f" ({end - start:.2f} seconds)")


with get_openai_callback() as cb2:
    start = time.time()
    result = chain.invoke({"input_documents": docs})
    end = time.time()
    print("--- result2")
    print(result["output_text"])
    print(str(cb2) + f" ({end - start:.2f} seconds)")
