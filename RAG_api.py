import streamlit as st
import json
import os
from typing import List
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load environment variables from .env

# --- Groq LLM Wrapper with Streaming ---
class GroqLLM(LLM):
    model: str = "llama-3.3-70b-versatile"  # Change as needed

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None) -> str:
        try:
            client = Groq()  # API key loaded from environment
            # Enable streaming
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=2,
                max_completion_tokens=512,
                top_p=1,
                stream=True,
                stop=stop,
            )
            # Stream tokens incrementally
            output = ""
            placeholder = st.empty()
            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                output += token
                placeholder.text(output)
            return output
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return f"Error: {str(e)}"

# --- Query Translation using GroqLLM ---
def translate_query(user_query: str) -> str:
    translation_prompt = (
        "You are a query clarifier for KIET College's information system. Your task:\n"
        "1. PRESERVE THE ORIGINAL QUERY if it is a greeting or already specific to KIET College and maintains correct meaning/grammer.\n"
        "2. REWRITE the query only when ambiguous references like 'your college' are present, replacing them with 'KIET College' or query is not written correctly.\n\n"
        f"Process this query: \"{user_query}\"\n"
        "Output ONLY the final query:"
    )
    llm = GroqLLM()
    translated = llm(translation_prompt)
    return translated.strip()

# --- Custom filter for similarity score threshold ---
def filter_by_threshold(vectorstore, query, k=6, threshold=0.4):
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    filtered_docs = []
    for doc, score in docs_and_scores:
        similarity = 1 - score / 2  # Adjust conversion as needed
        if similarity >= threshold:
            filtered_docs.append(doc)
    return filtered_docs

# --- Load and Process JSON Data ---
@st.cache_resource
def load_vectorstore():
    JSON_FILE_PATH = "extended_embeddings.json"
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
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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
    # --- Define Static Instructions Separately ---
    instructions = (
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
    )

    # --- Updated Prompt Template with Separated Retrieved Information ---
    prompt_template = PromptTemplate(
        input_variables=["instructions", "retrieved_info", "history", "question"],
        template=(
            "{instructions}\n\n"
            "Retrieved Information:\n"
            "{retrieved_info}\n\n"
            "Full Conversation History:\n"
            "{history}\n\n"
            "Caller's Query: {question}\n\n"
            "Phone-Optimized Response:"
        )
    )

    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Enter your query:", key="query_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_query:
        # Step 1: Translate the user query for clarity.
        translated_query = translate_query(user_query)
        st.info(f"Translated Query: {translated_query}")

        # Step 2: Retrieve relevant documents using the translated query.
        relevant_docs = filter_by_threshold(vectorstore, translated_query, k=5)
        retrieved_info = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

        # Store user query along with its retrieval data in chat history.
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "retrieved_info": retrieved_info
        })

        # Build full conversation history including only the last 8 messages.
        history = ""
        for msg in st.session_state.chat_history[-8:]:
            if msg["role"] == "user":
                history += f"User: {msg['content']}\n"
                if "retrieved_info" in msg and msg["retrieved_info"]:
                    history += f"Retrieved Info: {msg['retrieved_info']}\n"
            else:
                history += f"Assistant: {msg['content']}\n"

        # Step 3: Build the final prompt.
        prompt = prompt_template.format(
            instructions=instructions,
            retrieved_info=retrieved_info,
            history=history,
            question=translated_query
        )
        with st.expander("Complete Model Prompt", expanded=True):
            st.code(prompt, language="text")

        # Step 4: Call the Groq LLM (using streaming output).
        try:
            llm = GroqLLM()
            response = llm(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

    # Display the conversation history.
    for message in st.session_state.chat_history:
        st.write(f"**{message['role'].capitalize()}:** {message['content']}")
else:
    st.error("Failed to load the vector database. Please check your JSON file and try again.")
