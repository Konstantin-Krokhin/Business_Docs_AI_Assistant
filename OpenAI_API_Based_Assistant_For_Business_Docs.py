# Updated imports (replace all deprecated ones with these)
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings # If ran out of Open AI API tokens / Privacy required, use the local HuggingFace
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
#from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category = FutureWarning)

# ==============================
# 🔧 Load Environment & Config
# ==============================
#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_DIR = "./docs"
VECTOR_STORE_DIR = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# use a common compact model
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Stage 1: Load and Vectorize PDFs ===
def load_documents(DATA_DIR):
	loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
	docs = loader.load()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = 500,
		chunk_overlap = 100,
		separators = ["\n\n", "\n", ".", " "]
	)
	chunks = text_splitter.split_documents(docs)
	return chunks

def create_vectorstore(chunks):
	embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
	vectorstore = FAISS.from_documents(chunks, embeddings)
	vectorstore.save_local(VECTOR_STORE_DIR)
	return vectorstore

# === Stage 2: Local Hugging Face LLM Setup ===
def get_local_llm():
	local_pipeline = pipeline(
		task = "text2text-generation",
		model = "google/flan-t5-small", # Use small for CPU, change to Falcon if on GPU
		tokenizer = "google/flan-t5-small",
		max_new_tokens = 300,
		device = -1,  # -1 for CPU, 0 for GPU
	)
	return HuggingFacePipeline(pipeline = local_pipeline)

# === Stage 3: RetrievalQA Chain with Citations ===
def create_rag_chain(vectorstore, llm):
	retriever = vectorstore.as_retriever()
	qa_chain = RetrievalQA.from_chain_type(
		llm = llm,
		retriever = retriever,
		return_source_documents = True,
	)
	return qa_chain

# === Stage 4: Add New Docs Without Reprocessing ===
def add_new_documents(new_docs):
	loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
	new_chunks = loader.load()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = 500,
		chunk_overlap = 100,
		separators = ["\n\n", "\n", ".", " "]
	)
	new_chunks = text_splitter.split_documents(new_chunks)
	vectorstore.add_documents(new_chunks)
	vectorstore.save_local(VECTOR_STORE_DIR)

# === Stage 5: CLI Interaction ===
def run_cli_app():
	if os.path.exists(VECTOR_STORE_DIR):
		embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
		vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization = True)
	else:
		chunks = load_documents()
		vectorstore = create_vectorstore(chunks)

	llm  = get_local_llm()
	qa_chain = create_rag_chain(vectorstore, llm)

	print("\n🔍 Ask a question about your business docs (type 'exit' to quit):\n")
	while True:
		query = input("\nYou: ")
		if query.lower() == "exit":
			break

		result = qa_chain.invoke({"query": query})
		print(f"\n🤖 Answer:\n", result["result"])

if __name__ == "__main__":
	run_cli_app()