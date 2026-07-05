import streamlit as st
from pathlib import Path
from ingest import ingest_pdf, PDF_DIR  # reuse your existing constant
from query import ask, ask_stream, retrieve

st.set_page_config(page_title="PSH-01", layout="wide")
st.title("PSH-01: PENS Study Hub")

upload_tab, chat_tab = st.tabs(["📄 Upload", "💬 Chat"])

# --- Upload tab ---
with upload_tab:
    st.subheader("Add a document")
    uploaded_files = st.file_uploader("Drop PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Ingest"):
        for uploaded_file in uploaded_files:
            save_path = PDF_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Chunking and embedding {uploaded_file.name}..."):
                try:
                    ingest_pdf(save_path)
                    st.success(f"Ingested {uploaded_file.name}")
                except Exception as e:
                    st.error(f"{uploaded_file.name} failed: {e}")


# --- Chat tab ---
with chat_tab:
    st.subheader("Ask your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Sources"):
                    for chunk in msg["sources"]:
                        st.caption(f"{chunk.payload['filename']} — p.{chunk.payload['page']} (score: {chunk.score:.3f})")

    if question := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks = retrieve(question)          # get sources BEFORE streaming starts
                    full_answer = st.write_stream(ask_stream(question, chunks=chunks))

                    with st.expander("Sources"):
                        for chunk in chunks:
                            st.caption(f"{chunk.payload['filename']} — p.{chunk.payload['page']} (score: {chunk.score:.3f})")

                    st.session_state.messages.append({"role": "assistant", "content": full_answer, "sources": chunks})
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")