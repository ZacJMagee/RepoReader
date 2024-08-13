import os
import logging
import time
import re
import sys
import tempfile
import uuid
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
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
import sqlite3

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-3-5-sonnet-20240620"

logging.basicConfig(level=logging.INFO)

def sanitize_collection_name(name):
    # Remove any character that's not alphanumeric, underscore, or hyphen
    name = re.sub(r'[^\w\-]', '_', name)
    
    # Ensure the name starts and ends with an alphanumeric character
    name = re.sub(r'^[^\w]+', '', name)
    name = re.sub(r'[^\w]+$', '', name)
    
    # Replace consecutive underscores with a single underscore
    name = re.sub(r'_{2,}', '_', name)
    
    # Ensure the name is between 3 and 63 characters
    if len(name) < 3:
        name = name.ljust(3, 'a')
    if len(name) > 63:
        name = name[:63]
    
    # Ensure the name is not a valid IPv4 address
    if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', name):
        name = 'collection_' + name
    
    return name
def ensure_alternating_roles(conversation_history):
    processed_messages = []
    last_role = None
    for message in conversation_history:
        if message["role"] != last_role:
            processed_messages.append(message)
            last_role = message["role"]
    
    # Ensure the conversation starts with a user message
    if processed_messages and processed_messages[0]["role"] == "assistant":
        processed_messages.pop(0)
    
    return processed_messages

class DatabaseManager:
    def __init__(self, db_name="repositories.db"):
        self.db_name = db_name
        self.connect()

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                github_url TEXT NOT NULL UNIQUE,
                processed BOOLEAN NOT NULL DEFAULT 0
            )
        ''')
        self.conn.commit()

    def execute_with_retry(self, sql, params=None):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if params:
                    self.cursor.execute(sql, params)
                else:
                    self.cursor.execute(sql)
                self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    self.connect()  # Reconnect to the database
                else:
                    raise

    def add_repository(self, name, github_url):
        try:
            self.execute_with_retry('''
                INSERT INTO repositories (name, github_url) VALUES (?, ?)
            ''', (name, github_url))
            return True
        except sqlite3.IntegrityError:
            return False

    def get_all_repositories(self):
        self.execute_with_retry('SELECT * FROM repositories')
        return self.cursor.fetchall()

    def update_repository_status(self, github_url, processed):
        self.execute_with_retry('''
            UPDATE repositories SET processed = ? WHERE github_url = ?
        ''', (processed, github_url))

    def delete_repository(self, github_url):
        self.execute_with_retry('''
            DELETE FROM repositories WHERE github_url = ?
        ''', (github_url,))

    def close(self):
        self.conn.close()

class DocumentIndexer:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

    def index_documents(self, documents, cleanup_mode="incremental"):
        split_documents = self.split_and_prepare(documents)
        self.vector_store.add_documents(split_documents)

    def split_and_prepare(self, documents):
        split_documents = []
        for original_doc in documents:
            split_docs = self.text_splitter.split_documents([original_doc])
            split_documents.extend(split_docs)
        return split_documents

class RepositoryManager:
    def __init__(self, repo_path):
        self.repo_path = repo_path
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
            subprocess.run(["git", "clone", github_url, local_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to clone repository: {e}")
            return False

    def load_documents(self):
        documents = []
        for ext, LoaderClass in self.extensions.items():
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
                for doc in docs:
                    doc_id = str(uuid.uuid4())
                    doc.metadata["file_id"] = doc_id
                    documents.append(doc)
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
            py_docs = py_loader.load()
            for doc in py_docs:
                doc_id = str(uuid.uuid4())
                doc.metadata["file_id"] = doc_id
                documents.append(doc)
        except Exception as e:
            st.warning(f"Failed to load Python documents: {e}")

        return documents
class EnhancedDocumentRetriever:
    def __init__(self, collection_name, vectorstore=None):
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.store = InMemoryStore()
        
        if vectorstore is None:
            self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            self.vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embedding_model, persist_directory="./chroma_db")
        else:
            self.vectorstore = vectorstore
        
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            max_documents_per_query=5,
        )

    def add_documents(self, documents):
        self.retriever.add_documents(documents)

    def retrieve_documents(self, query):
        docs = self.retriever.invoke(query)
        logging.info(f"Retrieved {len(docs)} documents for query: {query}")
        
        # If no documents are found, try a more lenient search
        if len(docs) == 0:
            logging.info("No documents found. Attempting a more lenient search.")
            docs = self.vectorstore.similarity_search(query, k=5)
            logging.info(f"Lenient search retrieved {len(docs)} documents")
        
        # Log the content of retrieved documents
        for i, doc in enumerate(docs):
            logging.info(f"Document {i+1} content: {doc.page_content[:100]}...")  # Log first 100 chars
        
        return docs

class StreamlitApp:
    def __init__(self):
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.db_manager = DatabaseManager()

    def run(self):
        st.title("RAG Repository Analysis")

        # Initialize session state
        if 'repo_analyzed' not in st.session_state:
            st.session_state.repo_analyzed = False
        if 'document_retriever' not in st.session_state:
            st.session_state.document_retriever = None
        if 'repo_name' not in st.session_state:
            st.session_state.repo_name = ""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Add a sidebar for navigation and repository selection
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
        repositories = self.get_repositories()
        repo_options = ["Select a repository"] + [repo[1] for repo in repositories]
        selected_repo = st.sidebar.selectbox("Choose a repository", repo_options, key="repo_selector")

        if selected_repo != "Select a repository" and selected_repo != st.session_state.repo_name:
            st.session_state.repo_name = selected_repo
            st.session_state.repo_analyzed = True
            st.session_state.conversation_history = []
            self.setup_document_retriever_for_existing_repo(selected_repo)
            st.rerun()

    def handle_new_repo_input(self):
        github_url = st.text_input("Enter the GitHub URL of the repository:", key="github_url")
        if github_url:
            repo_name = github_url.split("/")[-1]
            existing_repo = next((repo for repo in self.db_manager.get_all_repositories() if repo[2] == github_url), None)
            
            if existing_repo:
                st.warning("This repository has already been added.")
                if st.button("Use existing repository"):
                    st.session_state.repo_name = existing_repo[1]
                    st.session_state.repo_analyzed = True
                    self.setup_document_retriever_for_existing_repo(existing_repo[1])
                    st.rerun()
                elif st.button("Reclone and reprocess repository"):
                    self.db_manager.delete_repository(github_url)
                    self.process_repository(github_url)
            else:
                if self.db_manager.add_repository(repo_name, github_url):
                    st.success(f"Repository {repo_name} added successfully!")
                    self.process_repository(github_url)

    @st.cache_data(ttl=60)  # Cache the result for 60 seconds
    def get_repositories(_self):
        return _self.db_manager.get_all_repositories()

    def reset_chat(self):
        st.session_state.repo_analyzed = False
        st.session_state.document_retriever = None
        st.session_state.repo_name = ""
        st.session_state.conversation_history = []
        st.rerun()

    def handle_repo_input(self):
        github_url = st.text_input("Enter the GitHub URL of the repository:", key="github_url")
        if github_url:
            repo_name = github_url.split("/")[-1]
            existing_repo = next((repo for repo in self.db_manager.get_all_repositories() if repo[2] == github_url), None)
            
            if existing_repo:
                st.warning("This repository has already been added.")
                if st.button("Use existing repository"):
                    st.session_state.repo_name = existing_repo[1]
                    st.session_state.repo_analyzed = True
                    self.setup_document_retriever_for_existing_repo(existing_repo[1])
                    st.rerun()
                elif st.button("Reclone and reprocess repository"):
                    self.db_manager.delete_repository(github_url)
                    self.process_repository(github_url)
            else:
                if self.db_manager.add_repository(repo_name, github_url):
                    st.success(f"Repository {repo_name} added successfully!")
                    self.process_repository(github_url)

    def setup_document_retriever_for_existing_repo(self, repo_name):
        """
        Set up the document retriever for an existing repository.
        This method loads the Chroma collection, creates the document retriever,
        and verifies the collection's contents.
        """
        sanitized_repo_name = sanitize_collection_name(repo_name)
        
        # Initialize the embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        
        # Load the existing Chroma collection
        vector_store = Chroma(
            collection_name=sanitized_repo_name,
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )
        
        # Create the document retriever using the loaded vector store
        st.session_state.document_retriever = EnhancedDocumentRetriever(
            collection_name=sanitized_repo_name,
            vectorstore=vector_store
        )
        
        # Verify the Chroma collection
        if self.verify_chroma_collection(repo_name):
            logging.info(f"Successfully loaded existing Chroma collection for {repo_name}")
        else:
            logging.warning(f"Chroma collection for {repo_name} is empty or not found")
        
        # Inspect the Chroma collection for detailed information
        self.inspect_chroma_collection(repo_name)

    def process_repository(self, github_url):
        with st.spinner("Cloning and indexing repository..."):
            with tempfile.TemporaryDirectory() as local_path:
                repository_manager = RepositoryManager(local_path)
                if repository_manager.clone_github_repo(github_url, local_path):
                    documents = repository_manager.load_documents()
                    if documents:
                        repo_name = github_url.split("/")[-1]
                        st.session_state.document_retriever = self.setup_document_retriever(documents, repo_name)
                        st.session_state.repo_name = repo_name
                        st.session_state.repo_analyzed = True
                        self.db_manager.update_repository_status(github_url, True)
                        st.success("Repository cloned and indexed successfully!")
                        st.rerun()
                    else:
                        st.error("No documents were found to index.")
                        st.session_state.document_retriever = None
                else:
                    st.error("Failed to clone the repository.")
                    st.session_state.document_retriever = None

    def handle_user_query(self):
        """
        Handle user queries about the repository.
        This method retrieves relevant documents, formats the context,
        and uses the LLM to generate a response.
        """
        st.write(f"Analyzing repository: {st.session_state.repo_name}")

        # Display conversation history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Get user input
        user_question = st.chat_input("Ask a question about the repository:")
        if user_question:
            if st.session_state.document_retriever is None:
                st.error("Document retriever is not set up. Please process a repository first.")
                return

            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                retrieved_docs = st.session_state.document_retriever.retrieve_documents(user_question)
                
                # Prepare context based on retrieved documents
                if not retrieved_docs:
                    st.warning("No relevant documents found for the query.")
                    context = f"No specific context available from the repository. The query is: {user_question}"
                else:
                    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])

                # Construct system message with context
                system_message = f"""You are a helpful assistant for analyzing the repository: {st.session_state.repo_name}. 
                Use the following context from the repository to answer the user's question. 
                If the context doesn't contain relevant information, say so and answer based on your general knowledge.
                Remember, you're analyzing a code repository, so focus on code-related aspects if possible.
                
                Repository Context:
                {context}
                """

                # Log the context being passed to the LLM
                logging.info(f"Context passed to LLM: {context[:500]}...")  # Log first 500 characters of context

                # Process conversation history to ensure alternating roles
                processed_history = ensure_alternating_roles(st.session_state.conversation_history[-10:])

                # Prepare messages for the LLM
                messages = [
                    {"role": "user", "content": f"User question: {user_question}"}
                ]
                messages.extend(processed_history)

                # Generate response using the LLM
                response = self.anthropic_client.messages.create(
                    model=MODEL_NAME,
                    system=system_message,
                    messages=messages,
                    max_tokens=1000
                )
                
                # Update conversation history
                st.session_state.conversation_history.append({"role": "user", "content": user_question})
                st.session_state.conversation_history.append({"role": "assistant", "content": response.content[0].text})

                # Display the latest response
                with st.chat_message("assistant"):
                    st.write(response.content[0].text)

    def setup_document_retriever(self, documents, repo_name):
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        sanitized_repo_name = sanitize_collection_name(repo_name)
        
        # Create or load the Chroma vector store with persistence
        vector_store = Chroma(
            collection_name=sanitized_repo_name,
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )
        
        # Create the document indexer
        document_indexer = DocumentIndexer(vector_store)
        
        # Index the documents
        document_indexer.index_documents(documents, cleanup_mode="incremental")
        
        # Added explicit persistence call for Chroma database
        vector_store.persist()
        logging.info(f"Persisted Chroma database for {repo_name}")
        
        # Create the document retriever with the populated vector store
        document_retriever = EnhancedDocumentRetriever(
            collection_name=sanitized_repo_name,
            vectorstore=vector_store
        )
        
        # Update the database to mark this repository as processed
        self.db_manager.update_repository_status(repo_name, processed=True)
        
        return document_retriever

    def verify_chroma_collection(self, repo_name):
        """
        Verify that the Chroma collection for the given repository exists and contains data.
        """
        sanitized_repo_name = sanitize_collection_name(repo_name)
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vector_store = Chroma(
            collection_name=sanitized_repo_name,
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )
        
        collection_size = vector_store.get()
        logging.info(f"Chroma collection '{sanitized_repo_name}' contains {len(collection_size['ids'])} documents")
        return len(collection_size['ids']) > 0

    def inspect_chroma_collection(self, repo_name):
        """
        Inspect and log detailed information about the Chroma collection for the given repository.
        """
        sanitized_repo_name = sanitize_collection_name(repo_name)
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vector_store = Chroma(
            collection_name=sanitized_repo_name,
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )
        
        collection = vector_store.get()
        logging.info(f"Inspecting Chroma collection '{sanitized_repo_name}':")
        logging.info(f"Number of documents: {len(collection['ids'])}")
        logging.info(f"Metadata keys: {list(collection['metadatas'][0].keys()) if collection['metadatas'] else 'No metadata'}")
        
        # Log a sample of document contents
        for i in range(min(5, len(collection['documents']))):
            logging.info(f"Sample document {i+1} content: {collection['documents'][i][:100]}...")  # Log first 100 chars

        return collection
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
