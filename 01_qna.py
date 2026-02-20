from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()
embeddings = OpenAIEmbeddings()

# loader = TextLoader('news/summary.txt')
loader = DirectoryLoader('news', glob="**/*.txt")

documents = loader.load()
print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# print(texts)

docsearch = Chroma.from_documents(texts, embeddings)
retriever = docsearch.as_retriever()
llm = OpenAI()

def query(q):
    print("Query: ", q)
    docs = retriever.invoke(q)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = llm.invoke(f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:")
    print("Answer: ", answer)

query("What are the effects of legislations surrounding emissions on the Australia coal market?")
query("What are China's plans with renewable energy?")
query("Is there an export ban on Coal in Indonesia? Why?")
query("Who are the main exporters of Coal to China? What is the role of Indonesia in this?")