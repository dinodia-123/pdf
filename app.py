import streamlit as st
import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import os

# Set OpenAI API key (Make sure to keep this secure in production)
openai.api_key = "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"  # <-- Insert your OpenAI key here

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks for manageable processing
def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to highlight relevant context in the original PDF
def highlight_relevant_text(context, question, chunks):
    relevant_chunks = []
    for chunk in chunks:
        if question.lower() in chunk.lower():  # Simple relevance check, could be more advanced
            relevant_chunks.append(chunk)
    return relevant_chunks

# Updated get_answer function
def get_answer(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",  # Use the specified model
        temperature=0,
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit UI Setup
st.set_page_config(page_title="📄 Chat with Your PDF", layout="wide")

st.title("📄 Chat with Your PDF")
st.markdown(
    """
    **Welcome to the Chat with PDF Application!**
    Upload your PDF document(s), and ask questions to extract relevant information.
    This application also highlights sections in the PDF relevant to your question.
    """
)

# Sidebar for PDF Upload
st.sidebar.header("Upload PDF(s)")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    all_chunks = []
    document_names = []
    
    for uploaded_file in uploaded_files:
        # Extract text from each PDF file
        text = extract_text_from_pdf(uploaded_file)
        document_names.append(uploaded_file.name)

        # Display the extracted content (preview)
        st.subheader(f"Extracted Content from {uploaded_file.name}")
        st.text_area(f"Content from {uploaded_file.name}:", text, height=300)
        
        # Split text into chunks for processing
        chunks = split_text(text)
        all_texts.append(text)
        all_chunks.append(chunks)
        
        # Show the number of chunks
        st.write(f"🔹 **{uploaded_file.name}** split into {len(chunks)} chunks for efficient processing.")

    # Ask a question
    st.subheader("Ask a Question")
    question = st.text_input("Type your question here:")

    if question:
        # Cross-document querying: Combine all chunks from different PDFs
        relevant_contexts = []
        for doc_index, chunks in enumerate(all_chunks):
            relevant_chunks = highlight_relevant_text(question, question, chunks)
            if relevant_chunks:
                relevant_contexts.extend(relevant_chunks)
        
        # Get the answer from OpenAI using the first relevant context found
        if relevant_contexts:
            context = " ".join(relevant_contexts)  # Combine relevant chunks into a single context
            with st.spinner("Generating answer..."):
                answer = get_answer(question, context)
            st.success("Answer:")
            st.write(answer)
        else:
            st.warning("No relevant content found in the uploaded documents for the given question.")
    
    # Optional: Display all chunks (debugging or transparency)
    with st.expander("Show all chunks (for debugging purposes)"):
        for i, chunks in enumerate(all_chunks):
            st.write(f"**Document: {document_names[i]}**")
            for i, chunk in enumerate(chunks):
                st.write(f"**Chunk {i+1}:** {chunk}")
else:
    st.info("Please upload one or more PDF files to get started.")
