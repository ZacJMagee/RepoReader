import os
import tempfile
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from file_processing import clone_github_repo, load_and_index_files, search_documents, format_documents, format_user_question
from questions import ask_question, QuestionContext

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"

class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames

def ask_question(question, context: QuestionContext):
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)

    numbered_documents = format_documents(relevant_docs)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}"

    answer_with_sources = context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context,
        repo_name=context.repo_name,
        github_url=context.github_url,
        conversation_history=context.conversation_history,
        numbered_documents=numbered_documents,
        file_type_counts=context.file_type_counts,
        filenames=context.filenames
    )
    return answer_with_sources

def main():
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                print("No documents were found to index. Exiting.")
                exit()

            print("Repository cloned. Indexing files...")
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

            # Create strings that will be used in the system message template
            file_type_counts_str = ', '.join(f"{k}: {v}" for k, v in file_type_counts.items())
            filenames_str = ', '.join(filenames)

            # Define the prompt with placeholders
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are a helpful assistant for analyzing code repositories. "
                           f"Repository Name: {repo_name}, GitHub URL: {github_url}, "
                           f"Document Count: {len(documents)}, File Type Counts: {file_type_counts_str}, "
                           f"File Names: {filenames_str}"),
                ("human", "{user_question}")
            ])
            chain = prompt | llm

            conversation_history = ""
            while True:
                user_question = input("\nAsk a question about the repository (type 'exit()' to quit): ")
                if user_question.lower() == 'exit()':
                    break
                print('Thinking...')

                # Invoke with a dictionary that matches the variable names used in the template
                full_response = chain.invoke({"user_question": user_question})

                # Properly accessing the AIMessage object and printing the entire object
                print(f'\nFULL RESPONSE\n{full_response}\n')
                conversation_history += f"Question: {user_question}\nFull Response: {full_response}\n"
        else:
            print("Failed to clone the repository.")
