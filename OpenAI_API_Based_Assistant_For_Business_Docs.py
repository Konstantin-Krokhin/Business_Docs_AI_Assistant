# Updated imports (replace all deprecated ones with these)
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
import os

# use a common compact model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()

# Make sure your OpenAI API key is set
#import os
# Make sure to set this outside the code, e.g., from terminal or .env
#api_key = os.environ["OPENAI_API_KEY"]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

#embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("local_vector_index")

retriever = vectorstore.as_retriever()

docs = retriever.invoke("What is travel reimbursement policy?")
for doc in docs:
    print(doc.page_content)