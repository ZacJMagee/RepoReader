import logging
import time
import threading
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import sqlite3
from utils import UtilityFunctions

class DatabaseManager:
    _local = threading.local()

    def __init__(self, db_name="repositories.db"):
        self.db_name = db_name

    def get_connection(self):
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_name)
            self._local.cursor = self._local.connection.cursor()
            self.create_table()
        return self._local.connection, self._local.cursor

    def create_table(self):
        conn, cursor = self.get_connection()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                github_url TEXT NOT NULL UNIQUE,
                processed BOOLEAN NOT NULL DEFAULT 0
            )
        ''')
        conn.commit()

    def execute_with_retry(self, sql, params=None):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                conn, cursor = self.get_connection()
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                conn.commit()
                return cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    self._local.connection = None  # Force a new connection on next attempt
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
        return self.execute_with_retry('SELECT * FROM repositories')

    def update_repository_status(self, github_url, processed):
        self.execute_with_retry('''
            UPDATE repositories SET processed = ? WHERE github_url = ?
        ''', (processed, github_url))

    def delete_repository(self, github_url):
        self.execute_with_retry('''
            DELETE FROM repositories WHERE github_url = ?
        ''', (github_url,))

    def close(self):
        if hasattr(self._local, 'connection') and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None
            self._local.cursor = None

    def verify_chroma_collection(self, repo_name):
        sanitized_repo_name = UtilityFunctions.sanitize_collection_name(repo_name)
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
        sanitized_repo_name = UtilityFunctions.sanitize_collection_name(repo_name)
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

    def __del__(self):
        self.close()
