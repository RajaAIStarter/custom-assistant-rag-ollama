import streamlit as st
import json
import subprocess
import os
from typing import List
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

# Configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
JSON_FILE_PATH = "extended_embeddings.json"
K_VALUE = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.4  # Minimum similarity score required

# --- Custom LLM Wrapper for Ollama with improved error handling ---
class OllamaLLM(LLM):
    model_name: str = OLLAMA_MODEL

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop=None) -> str:
        try:
            command = ["ollama", "run", self.model_name, prompt]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
            output = ""
            response_placeholder = st.empty()
            while process.poll() is None:
                char = process.stdout.read(1)
                if char:
                    output += char
                    if len(output) % 10 == 0:
                        response_placeholder.text(output)
            remaining_output = process.stdout.read()
            output += remaining_output
            response_placeholder.text(output)
            # Clear the streaming placeholder so the answer isn't duplicated in the chat window
            response_placeholder.empty()
            if process.returncode != 0:
                error = process.stderr.read()
                st.error(f"Ollama error: {error}")
                return f"Error running Ollama: {error}"
            return output
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return f"An error occurred: {str(e)}"

# --- Query Translation ---
def translate_query(user_query: str) -> str:
    translation_prompt = (
        "You are a query clarifier for KIET College's information system. Your task:\n"
        "1. PRESERVE THE ORIGINAL QUERY if it is a greeting or already specific to KIET College and maintains correct meaning/grammer.\n"
        "2. REWRITE the query only when ambiguous references like 'your college' are present, replacing them with 'KIET College' or query is not written correctly.\n\n"
        f"Process this query: \"{user_query}\"\n"
        "Output ONLY the final query with no additional text:"
    )
    llm = OllamaLLM()
    translated = llm(translation_prompt)
    return translated.strip()

# --- Custom filter for similarity score threshold ---
def filter_by_threshold(vectorstore, query, k=K_VALUE, threshold=SIMILARITY_THRESHOLD):
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    filtered_docs = []
    for doc, score in docs_and_scores:
        # Convert the FAISS distance to a similarity score (assuming 0 to 2 range)
        similarity = 1 - score / 2
        if similarity >= threshold:
            filtered_docs.append(doc)
    return filtered_docs

# --- Load and Process JSON Data with proper error handling ---
@st.cache_resource
def load_vectorstore():
    try:
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs: List[Document] = []

            def process_data(d, prefix=""):
                if isinstance(d, dict):
                    for k, v in d.items():
                        process_data(v, prefix + k + ": ")
                elif isinstance(d, list):
                    for item in d:
                        process_data(item, prefix)
                elif isinstance(d, str):
                    docs.append(Document(page_content=prefix + d))

            process_data(data)
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = FAISS.from_documents(docs, embedding_model)
            return vectorstore, len(docs)
        else:
            st.error(f"File not found: {JSON_FILE_PATH}")
            return None, 0
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {JSON_FILE_PATH}")
        return None, 0
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0

# --- Initialize Streamlit App ---
st.title("KIET College Caller Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

vectorstore, _ = load_vectorstore()

if vectorstore is not None:
    # Updated prompt template with conversation history included.
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "history"],
        template=(
            "Role: you are KIET College Call Assistant (for KIET, KIET Plus, KIET Womens), developed by KIET institutions.\n\n"
            "Instruction: Simulate a real phone conversation representing KIET College (Kakinada Institute of Engineering & Technology). "
            "Your responses must be formal, precise, and concise, using 3–4 sentences (10–40 words each). Maintain a courteous and friendly tone throughout the call.\n\n"
            "Protocol:\n"
            "- Open: 'Thank you for calling KIET!(only at first)'\n"
            "- Tone: Formal, concise and simulate an authentic phone conversation (15–50 words per response).\n"
            "- Close: 'Any other KIET queries before we end?'\n"
            "Guidelines:\n"
            "- Use retrieved data exclusively if available.\n"
            "- If no relevant data is retrieved and the query is basic or college-related, rely on your general knowledge about KIET College.\n"
            "- For off-topic, sensitive, or illegal queries, respond with: 'I don't have answers. Please ask me about KIET programs/services.'\n"
            "- Avoid negative or sensitive remarks about KIET or any topic."
            "Retrieved Information:\n"
            "{context}\n"
            "--- End of Retrieved Information ---\n\n"
            "Previous Conversation (last 4 exchanges):\n"
            "{history}\n"
            "--- End of Previous Conversation ---\n\n"
            "Caller's Query: {question}\n"
            "Phone-Optimized Response:"
        )

    )
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Enter your query:", key="query_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_query:
        # Step 1: Translate the user query for clarity.
        translated_query = translate_query(user_query)

        # Display the translated query for information.
        st.info(f"Translated Query: {translated_query}")

        # Add the original user query to the chat history.
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Step 2: Retrieve relevant documents using the translated query.
        relevant_docs = filter_by_threshold(vectorstore, translated_query, k=K_VALUE)
        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs])
        else:
            context = "No relevant information retrieved from the database."

        # Build conversation history from the last 4 exchanges (or fewer if not available)
        history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.chat_history[-8:]
        )

        # Step 3: Build the final prompt using the translated query and conversation history.
        prompt = prompt_template.format(context=context, question=translated_query, history=history)

        # Display the complete prompt for transparency.
        with st.expander("Complete Model Prompt", expanded=True):
            st.code(prompt, language="text")

        # Step 4: Call the LLM to generate the response.
        try:
            llm = OllamaLLM()
            response = llm(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

    # Display the chat history as a conversation-like window.
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        st.write(f"**{role.capitalize()}:** {content}")
else:
    st.error("Failed to load the vector database. Please check your JSON file and try again.")
