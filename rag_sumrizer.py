import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
try:
    import torch
except ImportError:
    torch = None

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("vector_data.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Device Selection
# -----------------------------------------------------------------------------
def select_device() -> str:
    """Select the optimal device for model inference."""
    if torch is not None:
        if torch.cuda.is_available():
            logger.info("Using CUDA device for embeddings.")
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("Using MPS device for embeddings.")
            return "mps"
    logger.info("Using CPU device for embeddings.")
    return "cpu"


# -----------------------------------------------------------------------------
# VectorData Class
# -----------------------------------------------------------------------------
class VectorData:
    """Manages document loading, embedding, and querying with ChromaDB."""

    def __init__(self, db_path: str = "db", collection_name: str = "legal_docs") -> None:
        """Initialize embedding model, device, and ChromaDB client."""
        try:
            device = select_device()
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(collection_name)
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            logger.info("VectorData initialized with model on %s.", device)
        except Exception as exc:
            logger.exception("Initialization failed: %s", exc)
            raise RuntimeError("VectorData initialization error") from exc

    # -------------------------------------------------------------------------
   

    # -------------------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar text chunks."""
        try:
            query_embedding = self.model.encode(query_text).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )

            if not results or not results.get("ids"):
                logger.warning("No results found for query: '%s'", query_text)
                return []

            matches = [
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results.get("distances", [[None]])[0][i],
                }
                for i in range(len(results["ids"][0]))
            ]

            logger.info("Query returned %d results.", len(matches))
            return matches
        except Exception as exc:
            logger.exception("Query error: %s", exc)
            return []

class ConversationManager:
    """
    Handles chatbot responses using context from the vector database and
    maintains optional conversation history.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the LLM and vector database connection."""
        try:
            self.llm = ChatGroq(
                api_key=api_key,
                model_name="openai/gpt-oss-20b",
                temperature=0.5,
                max_tokens=1024,
            )
            self.database = VectorData()
            logger.info("ConversationManager initialized successfully.")
        except Exception as exc:
            logger.exception("Failed to initialize ConversationManager: %s", exc)
            raise

    def response(self, query: str =None,transctipt: Any = None) -> str:
        """
        Generate a conversational response based on a query and optional history.
        Logs the entire lifecycle: query received → context retrieved → answer generated.
        """
        try:

            

            # Retrieve relevant context
            context = self.database.query(query_text=query)+self.database.query(transctipt)
            logger.info("Retrieved %d context items for query.", len(context))

            # Compose the prompt
            prompt = (
                f"Use the following context to answer the query.\n"
                f"Context: {context}\n"
                f"transcipt: {transctipt}\n"
                f"Generate a helpful, general, and intelligent answer."
            )

            # Invoke the model
            logger.debug("Invoking LLM for response generation.")
            ans = self.llm.invoke(prompt)

            logger.info("Response generated successfully for query: %s", query)
            return ans.content

        except Exception as exc:
            logger.exception("Error generating response for query '%s': %s", query, exc)
            return "An internal error occurred while processing your request."
# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


cloud_transcript = """
Cloud computing is the practice of using remote servers hosted on the internet 
to store, manage, and process data, instead of local servers or personal computers.

Key Points:
1. Scalability  resources can be scaled up or down based on demand.
2. Cost-efficiency pay only for what you use, no upfront hardware cost.
3. Accessibility  access data and services from anywhere with an internet connection.
4. Maintenance  providers handle updates, security, and uptime.
5. Examples  AWS, Google Cloud, Microsoft Azure.

In short, cloud computing transforms how we use and deliver IT resources, 
offering flexibility, reliability, and lower operational costs.
"""
cm=ConversationManager(api_key=)
print(cm.response(transctipt=cloud_transcript))