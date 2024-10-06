import os
from datetime import datetime
import logging
import tempfile
import uuid
import subprocess
import streamlit as st
from streamlit import cache_data, cache_resource
from dotenv import load_dotenv
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders.python import PythonLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from anthropic import Anthropic
from database import DatabaseManager
from utils import UtilityFunctions

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-3-5-sonnet-20240620"

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate a timestamp for the log file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/rag_app_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create a logger for the application
logger = logging.getLogger("RAGApp")

class CachedResources:
    @staticmethod
    @cache_resource
    def get_embedding_model():
        return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    @staticmethod
    @cache_resource
    def get_anthropic_client():
        return Anthropic(api_key=ANTHROPIC_API_KEY)

    @staticmethod
    @cache_resource
    def get_database_manager():
        return DatabaseManager()

    @staticmethod
    @cache_resource
    def get_vector_store(repo_name, _embedding_model):
        sanitized_repo_name = UtilityFunctions.sanitize_collection_name(repo_name)
        return Chroma(
            collection_name=sanitized_repo_name,
            embedding_function=_embedding_model,
            persist_directory="./chroma_db"
        )

    @staticmethod
    @cache_resource
    def get_document_retriever(collection_name, _vectorstore):
        return EnhancedDocumentRetriever(collection_name, _vectorstore)

    @staticmethod
    @cache_data(ttl=60)
    def get_repositories(_db_manager):
        return _db_manager.get_all_repositories()

class RepositoryManager:
    def __init__(self, repo_path, vector_store):
        self.repo_path = repo_path
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self.extensions = {
            "txt": TextLoader,
            "md": TextLoader,
            "markdown": TextLoader,
            "rst": TextLoader,
            "js": TextLoader,
            "java": TextLoader,
            "c": TextLoader,
            "cpp": TextLoader,
            "cs": TextLoader,
            "go": TextLoader,
            "rb": TextLoader,
            "php": TextLoader,
            "scala": TextLoader,
            "html": TextLoader,
            "htm": TextLoader,
            "xml": TextLoader,
            "json": TextLoader,
            "yaml": TextLoader,
            "yml": TextLoader,
            "ini": TextLoader,
            "toml": TextLoader,
            "cfg": TextLoader,
            "conf": TextLoader,
            "sh": TextLoader,
            "bash": TextLoader,
            "css": TextLoader,
            "scss": TextLoader,
            "sql": TextLoader,
            "ipynb": NotebookLoader,
        }

    def clone_github_repo(self, github_url, local_path):
        try:
            logger.info(f"Cloning repository: {github_url} to {local_path}")
            subprocess.run(["git", "clone", github_url, local_path], check=True)
            logger.info(f"Repository cloned successfully: to {local_path}")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to clone repository: {e}")
            return False

    def load_documents(self):
        documents = []
        for ext, LoaderClass in self.extensions.items():
            logger.info(f"Loading documents from {self.repo_path}")
            loader = DirectoryLoader(
                self.repo_path,
                glob=f"**/*.{ext}",
                loader_cls=LoaderClass,
                loader_kwargs={"autodetect_encoding": True}
                if LoaderClass != PythonLoader
                else {},
                use_multithreading=True,
                show_progress=True,
            )
            try:
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} documents for {ext}")
                for doc in docs:
                    doc_id = str(uuid.uuid4())
                    doc.metadata["file_id"] = doc_id
                    documents.append(doc)
                    logger.info(f"Total documents loaded: {len(documents)}")
            except Exception as e:
                st.warning(f"Failed to load documents for {ext}: {e}")

        # Handle Python files separately
        py_loader = DirectoryLoader(
            self.repo_path,
            glob="**/*.py",
            loader_cls=PythonLoader,
            use_multithreading=True,
            show_progress=True,
        )
        try:
            logger.info(f"Loading Python documents from {self.repo_path}")
            py_docs = py_loader.load()
            for doc in py_docs:
                doc_id = str(uuid.uuid4())
                doc.metadata["file_id"] = doc_id
                documents.append(doc)
        except Exception as e:
            st.warning(f"Failed to load Python documents: {e}")

        return documents

    def index_documents(self, documents, cleanup_mode="incremental"):
        logger.info(f"Indexing {len(documents)} documents")
        split_documents = self.split_and_prepare(documents)
        self.vector_store.add_documents(split_documents)
        logger.info(f"Finished {len(documents)}indexing documents")

    def split_and_prepare(self, documents):
        split_documents = []
        for original_doc in documents:
            split_docs = self.text_splitter.split_documents([original_doc])
            split_documents.extend(split_docs)
        return split_documents

    def process_repository(self, github_url):
        if self.clone_github_repo(github_url, self.repo_path):
            documents = self.load_documents()
            if documents:
                self.index_documents(documents)
                return True
            else:
                return False
        return False

    
class EnhancedDocumentRetriever:
    def __init__(self, collection_name, vectorstore=None):
        logger.info(f"Initializing EnhancedDocumentRetriever with collection name: {collection_name}")
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.store = InMemoryStore()
        
        if vectorstore is None:
            logger.info("Creating new embedding model and vector store")
            self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            self.vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embedding_model, persist_directory="./chroma_db")
        else:
            logger.info("Using provided vector store")
            self.vectorstore = vectorstore
        
        logger.info("Setting up ParentDocumentRetriever")
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def add_documents(self, documents):
        logger.info(f"Adding {len(documents)} documents to the retriever")
        self.retriever.add_documents(documents)
        logger.info("Documents added successfully")

    def retrieve_documents(self, query):
        logger.info(f"Retrieving documents for query: {query}")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")
        
        if len(docs) == 0:
            logger.info("No documents found. Attempting a more lenient search.")
            docs = self.vectorstore.similarity_search(query, k=5)
            logger.info(f"Lenient search retrieved {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            logger.debug(f"Document {i+1} content (first 100 chars): {doc.page_content[:100]}...")
        
        return docs

class StreamlitApp:
    def __init__(self):
        logger.info("Initializing StreamlitApp")
        self.anthropic_client = CachedResources.get_anthropic_client()
        self.db_manager = CachedResources.get_database_manager()

    def run(self):
        logger.info("Starting StreamlitApp")
        st.title("RAG Repository Analysis")

        if 'repo_analyzed' not in st.session_state:
            st.session_state.repo_analyzed = False
        if 'document_retriever' not in st.session_state:
            st.session_state.document_retriever = None
        if 'repo_name' not in st.session_state:
            st.session_state.repo_name = ""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        st.sidebar.title("Repository Selection")
        
        option = st.sidebar.radio("Choose an option:", ["Select Existing Repository", "Add New Repository"])

        if option == "Select Existing Repository":
            self.handle_existing_repo_selection()
        else:
            self.handle_new_repo_input()

        if st.session_state.repo_analyzed:
            self.handle_user_query()

        if st.sidebar.button("Start New Chat"):
            self.reset_chat()

    def handle_existing_repo_selection(self):
        logger.info("Handling existing repository selection")
        repositories = CachedResources.get_repositories(self.db_manager)
        repo_options = ["Select a repository"] + [repo[1] for repo in repositories]
        selected_repo = st.sidebar.selectbox("Choose a repository", repo_options, key="repo_selector")

        if selected_repo != "Select a repository" and selected_repo != st.session_state.repo_name:
            logger.info(f"Selected repository: {selected_repo}")
            st.session_state.repo_name = selected_repo
            st.session_state.repo_analyzed = True
            st.session_state.conversation_history = []
            self.setup_document_retriever_for_existing_repo(selected_repo)
            st.rerun()

    def handle_new_repo_input(self):
        logger.info("Handling new repository input")
        github_url = st.text_input("Enter the GitHub URL of the repository:", key="github_url")
        if github_url:
            logger.info(f"Processing new repository: {github_url}")
            repo_name = github_url.split("/")[-1]
            existing_repo = next((repo for repo in self.db_manager.get_all_repositories() if repo[2] == github_url), None)
            
            if existing_repo:
                logger.info("Repository already exists in database")
                st.warning("This repository has already been added.")
                if st.button("Use existing repository"):
                    logger.info(f"Using existing repository: {existing_repo[1]}")
                    st.session_state.repo_name = existing_repo[1]
                    st.session_state.repo_analyzed = True
                    self.setup_document_retriever_for_existing_repo(existing_repo[1])
                    st.rerun()
                elif st.button("Reclone and reprocess repository"):
                    logger.info(f"Recloning and reprocessing repository: {github_url}")
                    self.db_manager.delete_repository(github_url)
                    self.process_repository(github_url)
            else:
                logger.info(f"Adding new repository: {repo_name}")
                if self.db_manager.add_repository(repo_name, github_url):
                    st.success(f"Repository {repo_name} added successfully!")
                    self.process_repository(github_url)

    @st.cache_data(ttl=60)
    def get_repositories(_self):
        logger.info("Fetching repositories from database")
        return _self.db_manager.get_all_repositories()

    def reset_chat(self):
        logger.info("Resetting chat")
        st.session_state.repo_analyzed = False
        st.session_state.document_retriever = None
        st.session_state.repo_name = ""
        st.session_state.conversation_history = []
        st.rerun()

    

    def setup_document_retriever_for_existing_repo(self, repo_name):
        logger.info(f"Setting up document retriever for existing repository: {repo_name}")
        sanitized_repo_name = UtilityFunctions.sanitize_collection_name(repo_name)
        
        logger.info("Creating embedding model")
        embedding_model = CachedResources.get_embedding_model()
        
        logger.info(f"Loading existing Chroma collection: {sanitized_repo_name}")
        vector_store = CachedResources.get_vector_store(repo_name, embedding_model)
        
        logger.info("Creating document retriever")
        st.session_state.document_retriever = CachedResources.get_document_retriever(sanitized_repo_name, vector_store)
        
        if self.db_manager.verify_chroma_collection(repo_name):
            logger.info(f"Successfully loaded existing Chroma collection for {repo_name}")
        else:
            logger.warning(f"Chroma collection for {repo_name} is empty or not found")
        
        logger.info(f"Inspecting Chroma collection for {repo_name}")
        self.db_manager.inspect_chroma_collection(repo_name)

    def process_repository(self, github_url):
        logger.info(f"Processing repository: {github_url}")
        with st.spinner("Cloning and indexing repository..."):
            with tempfile.TemporaryDirectory() as local_path:
                repo_name = github_url.split("/")[-1]
                sanitized_repo_name = UtilityFunctions.sanitize_collection_name(repo_name)
                logger.info(f"Creating vector store for {sanitized_repo_name}")
                vector_store = CachedResources.get_vector_store(repo_name, CachedResources.get_embedding_model())
                repository_manager = RepositoryManager(local_path, vector_store)
                if repository_manager.process_repository(github_url):
                    logger.info("Repository processed successfully")
                    st.session_state.document_retriever = CachedResources.get_document_retriever(sanitized_repo_name, vector_store)
                    st.session_state.repo_name = repo_name
                    st.session_state.repo_analyzed = True
                    st.success("Repository cloned and indexed successfully!")
                    st.rerun()
                else:
                    logger.error("Failed to process the repository")
                    st.error("Failed to process the repository.")
                    st.session_state.document_retriever = None

    def handle_user_query(self):
        logger.info(f"Handling user query for repository: {st.session_state.repo_name}")
        st.write(f"Analyzing repository: {st.session_state.repo_name}")

        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input("Ask a question about the repository:")
        if user_question:
            logger.info(f"User question: {user_question}")
            if st.session_state.document_retriever is None:
                logger.error("Document retriever is not set up")
                st.error("Document retriever is not set up. Please process a repository first.")
                return

            with st.spinner("Thinking..."):
                logger.info("Retrieving relevant documents")
                retrieved_docs = st.session_state.document_retriever.retrieve_documents(user_question)
                
                if not retrieved_docs:
                    logger.warning("No relevant documents found for the query")
                    st.warning("No relevant documents found for the query.")
                    context = f"No specific context available from the repository. The query is: {user_question}"
                else:
                    logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
                    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])

                system_message = f"""You are a helpful assistant for analyzing the repository: {st.session_state.repo_name}. 
                Use the following context from the repository to answer the user's question. 
                If the context doesn't contain relevant information, say so and answer based on your general knowledge.
                Remember, you're analyzing a code repository, so focus on code-related aspects if possible.
                
                Repository Context:
                {context}
                """
                logger.debug(f"System message: {system_message[:500]}...") # First 500 chars

                messages = []
                for i in range(0, len(st.session_state.conversation_history), 2):
                    if i < len(st.session_state.conversation_history):
                        messages.append({"role": "user", "content": st.session_state.conversation_history[i]["content"]})
                    if i + 1 < len(st.session_state.conversation_history):
                        messages.append({"role": "assistant", "content": "..."})
                
                messages.append({"role": "user", "content": user_question})

                logger.info(f"Sending query to LLM with {len(messages)} message(s) in history")
                response = self.anthropic_client.messages.create(
                    model=MODEL_NAME,
                    system=system_message,
                    messages=messages,
                    max_tokens=1000
                )
                logger.info("Received response from LLM")
                
                response_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                
                st.session_state.conversation_history.append({"role": "user", "content": user_question})
                st.session_state.conversation_history.append({"role": "assistant", "content": response_text})

                with st.chat_message("assistant"):
                    st.write(response_text)

            logger.info(f"Messages sent to API: {messages}")

    def setup_document_retriever(self, repo_name):
        logger.info(f"Setting up document retriever for repository: {repo_name}")
        sanitized_repo_name = UtilityFunctions.sanitize_collection_name(repo_name)
        
        logger.info(f"Creating/loading Chroma vector store for {sanitized_repo_name}")
        vector_store = CachedResources.get_vector_store(repo_name, CachedResources.get_embedding_model())
        
        logger.info("Creating document retriever")
        document_retriever = CachedResources.get_document_retriever(sanitized_repo_name, vector_store)
        
        logger.info(f"Updating repository status for {repo_name}")
        self.db_manager.update_repository_status(repo_name, processed=True)
        
        return document_retriever

if __name__ == "__main__":
    logger.info("Starting RAG Repository Analysis application")
    app = StreamlitApp()
    app.run()
