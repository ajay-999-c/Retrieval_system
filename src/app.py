import streamlit as st
from retriever import RetrieverSystem

# 1. Initialize Retriever
retriever_system = RetrieverSystem()

# 2. Streamlit UI
st.set_page_config(page_title="🔎 Bignalytics RAG Chatbot", page_icon="🔍")
st.title("🔎 Bignalytics FAQ Retriever with Section Filter")

query = st.text_input("Ask your question:")

# # Optional Section Filter
section_options = [
    "None", "About Institute", "Admission Process", "Certification", "Placement", 
    "Courses Offered", "Contact Information", "Eligibility", "Facilities", "Demo Classes",
    "Batch Information", "Loan/EMI Support", "Blogs and Articles"
]

selected_section = st.selectbox("Optional: Filter by Section", section_options)

k = st.slider("Select Top-K Results", min_value=1, max_value=10, value=3)
if st.button("Retrieve Answer"):
    if query.strip() != "":
        with st.spinner("Searching..."):
            # Clean filter
            section_filter = None if selected_section == "None" else selected_section

            docs = retriever_system.retrieve(query=query, k=k, section_filter=section_filter)
        
        if docs:
            for idx, doc in enumerate(docs):
                st.subheader(f"Result {idx+1}")
                st.markdown(f"**Section:** {doc.metadata.get('section', 'Unknown')}")
                st.markdown(f"**Original Question:** {doc.metadata.get('question', 'Unknown')}")
                st.markdown(f"**Answer:** \n{doc.page_content}")
        else:
            st.error("❗ No relevant answer found. Please try a different question.")
    else:
        st.warning("Please type a question to search!")
