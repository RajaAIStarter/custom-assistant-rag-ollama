import streamlit as st
import subprocess
import time
import json
import numpy as np
import re
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# --- Utility Functions ---

def load_embeddings(file_path):
    """
    Load precomputed embeddings and document chunks from a JSON file.
    The JSON should have keys "chunks" and "embeddings".
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])
    return chunks, embeddings

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def filter_chunks_by_keywords(chunks, query):
    """
    Filter the retrieved chunks by checking if they contain at least one non-trivial token from the query.
    """
    # Remove punctuation and convert to lowercase
    query_clean = re.sub(r'\W+', ' ', query.lower()).strip()
    query_tokens = query_clean.split()
    # Define a simple set of stopwords to ignore
    stopwords = {"the", "is", "a", "an", "and", "of", "in", "to", "for", "i", "you", "on", "that"}
    query_tokens = [token for token in query_tokens if token not in stopwords]

    filtered_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Keep chunk if at least one query token is present
        if any(token in chunk_lower for token in query_tokens):
            filtered_chunks.append(chunk)
    return filtered_chunks

def retrieve_chunks(query, model, chunks, embeddings, top_k=3, threshold=0.4):
    query_embedding = model.encode([query])[0]
    sims = [cosine_similarity(query_embedding, emb) for emb in embeddings]

    if not sims or max(sims) < threshold:
        return []

    # Get indices of chunks with similarity above threshold.
    indices = [i for i, sim in enumerate(sims) if sim >= threshold]
    if not indices:
        return []

    top_indices = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]
    retrieved = [chunks[i] for i in top_indices]
    # Apply additional keyword filtering.
    filtered = filter_chunks_by_keywords(retrieved, query)
    return filtered

def build_prompt(conversation_history, retrieved_chunks, query):
    header = (
        "Instruction: You are a knowledgeable and helpful caller assistant for KIET College, developed by them ,representing "
        "Kakinada Institute of Engineering and Technology, Kiet Plus, and KIET Womens. Your responses must be "
        "precise, formal, and simulate a real phone conversation. Answer college-related queries concisely (ideally between 10 to  40 words) "
        "and avoid extraneous details. If no relevant data is available, clearly state, 'I'm sorry, I don't have that information.' "
        "If a query deviates from college-related topics or involves sensitive/off-topic content, politely warn that only current, "
        "college-related information can be provided. Maintain a courteous, friendly tone with appropriate greetings and conversational cues.\n\n"
    )

    conversation = ""
    for chat in conversation_history:
        if chat["role"] == "user":
            conversation += "User: " + chat["message"] + "\n"
        else:
            conversation += "Assistant: " + chat["message"] + "\n"
    delimiter = "\n--- Retrieved Information ---\n"
    retrieved_text = "\n".join(retrieved_chunks) if retrieved_chunks else ""
    end_delimiter = "\n--- End of Retrieved Information ---\n"
    query_reminder = "Based on the above, answer the following question: " + query + "\n"
    prompt = header + conversation + delimiter + retrieved_text + end_delimiter + query_reminder
    return prompt

def stream_model_response(prompt):
    """
    Call the local model (via the 'ollama' command) and stream its response character-by-character.
    Adjust the command as needed.
    """
    command = ["ollama", "run", "qwen2.5:3b", prompt]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        yield char
    process.stdout.close()
    process.wait()

# --- Main App ---
def main():
    st.title("_____")

    # Initialize session state for chat history.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load precomputed embeddings and chunks.
    if "chunks" not in st.session_state or "embeddings" not in st.session_state:
        chunks, embeddings = load_embeddings("embeddings_extended.json")
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

    # Load the embedding model (same as used for embeddings generation).
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Display chat history using a chat-like interface if available.
    if hasattr(st, "chat_message"):
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["message"])
            else:
                st.chat_message("assistant").write(msg["message"])
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['message']}")
            else:
                st.markdown(f"**Assistant:** {msg['message']}")

    # Chat input form.
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your message:")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Append the user message to the chat history.
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        # Retrieve relevant chunks based on the query.
        retrieved_chunks = retrieve_chunks(user_input, model, st.session_state.chunks, st.session_state.embeddings,
                                           top_k=3, threshold=0.42)

        # Build the final prompt.
        final_prompt = build_prompt(st.session_state.chat_history, retrieved_chunks, user_input)

        # (Optional) Uncomment to display the final prompt for debugging:
        # st.write("DEBUG - Final Prompt:", final_prompt)

        # Stream the model's response.
        response_placeholder = st.empty()
        full_response = ""
        for char in stream_model_response(final_prompt):
            full_response += char
            response_placeholder.markdown(f"**Assistant:** {full_response}")
            time.sleep(0.0001)  # Optimized streaming delay for faster response.
        st.session_state.chat_history.append({"role": "assistant", "message": full_response})

if __name__ == "__main__":
    main()
