# Updated imports (replace all deprecated ones with these)
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
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

def detect_loader_by_extension(file_path):
	ext = os.path.splitext(file_path)[1].lower()
	if ext == ".pdf":
		return PyMuPDFLoader(file_path)
	elif ext in [".txt", ".md"]:
		return TextLoader(file_path, encoding = "utf-8")
	elif ext == ".docx":
		from langchain_community.document_loaders import Docx2txtLoader
		return Docx2txtLoader(file_path)
	elif ext == ".csv":
		from langchain_community.document_loaders import CSVLoader
		return CSVLoader(file_path, encoding = "utf-8")
	elif ext == ".json":
		from langchain_community.document_loaders import JSONLoader
		return JSONLoader(file_path, encoding = "utf-8")
	elif ext == ".xlsx":
		from langchain_community.document_loaders import UnstructuredExcelLoader
		return UnstructuredExcelLoader(file_path)
	elif ext == ".pptx":
		from langchain_community.document_loaders import UnstructuredPowerPointLoader
		return UnstructuredPowerPointLoader(file_path)
	elif ext == ".html":
		from langchain_community.document_loaders import UnstructuredHTMLLoader
		return UnstructuredHTMLLoader(file_path)
	elif ext == ".xml":
		from langchain_community.document_loaders import UnstructuredXMLLoader
		return UnstructuredXMLLoader(file_path)
	else:
		raise ValueError(f"Unsupported file type: {ext}. Supported types are .pdf, .txt, and .md.")

warnings.filterwarnings("ignore", category = FutureWarning)

def get_embeddings():
	return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

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
def load_and_chunk_documents(DATA_DIR):
	loader = DirectoryLoader(DATA_DIR, glob="**/*.*", loader_cls=detect_loader_by_extension)
	docs = loader.load()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = 500,
		chunk_overlap = 100,
		separators = ["\n\n", "\n", ".", " "]
	)
	chunks = text_splitter.split_documents(docs)
	return chunks

def create_vectorstore(chunks):
	embeddings = get_embeddings()
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
def add_new_documents(folder_path, vectorstore):
	loader = DirectoryLoader(folder_path, glob="**/*.*", loader_cls = detect_loader_by_extension)
	new_docs = loader.load()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = 500,
		chunk_overlap = 100,
		separators = ["\n\n", "\n", ".", " "]
	)
	new_chunks = text_splitter.split_documents(new_docs)
	vectorstore.add_documents(new_chunks)
	vectorstore.save_local(VECTOR_STORE_DIR)
	return vectorstore

# === Stage 5: CLI Interaction ===
def run_cli_app():
	if os.path.exists(VECTOR_STORE_DIR):
		embeddings = get_embeddings()
		vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization = True)
	else:
		chunks = load_and_chunk_documents(DATA_DIR)
		vectorstore = create_vectorstore(chunks)

	llm  = get_local_llm()
	qa_chain = create_rag_chain(vectorstore, llm)

	print("\n🔍 Ask a question about your business docs (type 'exit' to quit or 'add' to upload new files):\n")
	while True:
		query = input("\nYou: ")
		if query.lower() == "exit":
			break
		elif query.lower() == "add":
			print("Adding new documents...")
			folder_path = input("Enter the folder path containing new documents: ")
			if os.path.exists(folder_path):
				vectorstore = add_new_documents(folder_path, vectorstore)
				qa_chain = create_rag_chain(vectorstore, llm)
				print("✅ New documents added!")
			else:
				print(f"❌ Folder {folder_path} does not exist. Please check the path and try again.")
			continue

		result = qa_chain.invoke({"query": query})
		print(f"\n🤖 Answer:\n", result["result"])

	for i, doc in enumerate(result["source_documents"]):
		print(f"\n📄 Source Document {i + 1}:\n", doc.page_content[:500], "...\n")

if __name__ == "__main__":
	run_cli_app()