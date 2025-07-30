# Updated imports (replace all deprecated ones with these)
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings # If ran out of Open AI API tokens / Privacy required, use the local HuggingFace
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from lanchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category = FutureWarning)

# ==============================
# 🔧 Load Environment & Config
# ==============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_DIR = "./docs"
VECTOR_STORE_DIR = "./vectorstore"

# use a common compact model
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 256,
	chunk_overlap = 50,
	separators = ["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(docs)

if os.path.exists(VECTOR_STORE_DIR):
	vectorstore = FAISS.load_local(VECTOR_STORE_DIR, HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"))
else:
	vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"))
	vectorstore.save_local(VECTOR_STORE_DIR)


retriever = vectorstore.as_retriever(search_kwargs={"k" : 4})

llm = ChatOpenAI(
	model_name = "gpt-4o",
	temperature = 0
)

qa_chain = RetrievalQA.from_chain_type(
	llm = llm,
	retriever = retriever,
	return_source_documents = True,
)

print("\n🔍 Ask a question about your business docs (type 'exit' to quit):\n")
while True:
	query = input("\nYou: ")
	if query.lower() == "exit":
		break

result = qa_chain.invoke({"query" : query})
answer = result["result"]
source_docs = result["source_documents"]

print(f"\n🤖 Answer: {answer}\n")

