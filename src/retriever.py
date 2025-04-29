from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from logger import log_event  # Assumes logger.py with log_event() is available


class RetrieverSystem:
    """
    RAG Retriever System using HuggingFace Embeddings and Chroma Vector Store.
    
    Loads an existing vectorstore from disk, prepares retriever, and retrieves relevant documents.
    """

    def __init__(self, persist_directory: str = "./chroma_db", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever system.

        Args:
            persist_directory (str): Path to persisted Chroma vector store.
            embedding_model_name (str): HuggingFace embedding model to use.
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.vectorstore = None
        self.retriever = None

        log_event("🔄 Initializing RetrieverSystem...")
        self._load_vectorstore()
        log_event("✅ RetrieverSystem initialization complete.\n")

    def _load_vectorstore(self):
        """
        Load the Chroma vector store and initialize the retriever.
        """
        log_event(f"📂 Loading Chroma vectorstore from: {self.persist_directory}")
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding_model
        )
        self.retriever = self.vectorstore.as_retriever()
        log_event(f"✅ Chroma vectorstore loaded and retriever initialized.\n")

    def retrieve(self, query: str, k: int = 5, section_filter: str = None):
        """
        Retrieve top-k most relevant documents from the vectorstore.

        Args:
            query (str): User query.
            k (int): Number of top documents to return.
            section_filter (str, optional): Optional metadata filter based on 'section' key.

        Returns:
            List[Document]: Top-k retrieved LangChain Document objects.
        """
        log_event(f"🔎 Retrieving for query: '{query}' | Top-K: {k} | Filter: {section_filter or 'None'}")

        if section_filter:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter={"section": section_filter}
            )
            log_event(f"✅ Retrieved {len(docs)} documents with section filter '{section_filter}'.\n")
        else:
            docs = self.vectorstore.similarity_search(query=query, k=k)  
            log_event(f"✅ Retrieved {len(docs)} documents with no filter.\n")

        return docs
