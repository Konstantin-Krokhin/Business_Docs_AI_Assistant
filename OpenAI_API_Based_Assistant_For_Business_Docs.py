from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader  # For plain .txt files
from langchain_community.document_loaders import PyPDFLoader  # For PDFs (no layout model)
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

# use a common compact model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()

# Make sure your OpenAI API key is set
#import os
#os.environ['OPENAI_API_KEY'] = "[REMOVED_SECRET]"

# Rest of your code is correct
splitter = RecursiveCharacterTextSplitter(chunk_size=598, chunk_overlap=198)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
db.save_local('/Kaggle/working/vector_index') # Explicit Kaggle path