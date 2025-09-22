import os
import json
import requests
from typing import List, Dict, Any, Optional
import re
from processFile import ProcessFile


# External deps:
#   pip install chonkie pinecone-client
from chonkie import SemanticChunker
from pinecone import Pinecone

mapa = {
  'á':'a','Á':'A','à':'a','À':'A','ä':'a','Ä':'A','â':'a','Â':'A','ã':'a','Ã':'A',
  'é':'e','É':'E','è':'e','È':'E','ë':'e','Ë':'E','ê':'e','Ê':'E',
  'í':'i','Í':'I','ì':'i','Ì':'I','ï':'i','Ï':'I','î':'i','Î':'I',
  'ó':'o','Ó':'O','ò':'o','Ò':'O','ö':'o','Ö':'O','ô':'o','Ô':'O','õ':'o','Õ':'O',
  'ú':'u','Ú':'U','ù':'u','Ù':'U','ü':'u','Ü':'U','û':'u','Û':'U',
  'ñ':'n','Ñ':'N'
}


pattern = re.compile(r'[áÁàÀäÄâÂãÃéÉèÈëËêÊíÍìÌïÏîÎóÓòÒöÖôÔõÕúÚùÙüÜûÛñÑ]')

def reemplazar(match):
    return mapa.get(match.group(0), match.group(0))

def quitar_acentos_regex(s):
    return pattern.sub(reemplazar, s)

class OllamaRAGSystem:
    def __init__(self, pinecone_api_key: str, pinecone_index: str, ollama_model: str = "llama2:7b-chat"):
        """
        Initialize the RAG system

        Args:
            pinecone_api_key: Your Pinecone API key
            pinecone_index: Name of your Pinecone index
            ollama_model: Ollama model to use for generation
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index)

        # Ollama configuration
        self.ollama_model = ollama_model
        # self.ollama_url = "http://localhost:11434/api/chat"
        self.ollama_url = "http://localhost:11434/api/generate"

        # Semantic chunker configuration
        self.chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.7,
            chunk_size=512
        )

    def load_texts(self, folder_path: str) -> List[Dict]:
        documents: List[Dict[str, Any]] = []
        print(f"Scanning folder: {folder_path}")

        for filename in os.listdir(folder_path):
            print('filename to scan', filename)
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                print(f"\nProcessing document: {filename}")

                try:
                    text = self.load_document(file_path)
                    text = quitar_acentos_regex(text)

                    doc_info = {
                        "filename": filename,
                        "path": file_path,
                        "content": text,
                        "length_chars": len(text),
                        "length_words": len(text.split())
                    }
                    documents.append(doc_info)
                    print(f"✓ Loaded {doc_info['length_chars']} chars, {doc_info['length_words']} words")
                except Exception as e:
                    print(f"⚠ Skipped {filename} (Error: {e})")

        print(f"\nTotal documents loaded: {len(documents)}")
        return documents

    def load_markdowns(self, folder_path: str) -> List[Dict]:
        """
        Load all markdown files in the folder.

        Returns:
            A list of dictionaries with file metadata and content.
        """
        documents: List[Dict[str, Any]] = []
        print(f"Scanning folder: {folder_path}")

        for filename in os.listdir(folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(folder_path, filename)
                print(f"\nProcessing document: {filename}")

                try:
                    text = self.load_document(file_path)
                    doc_info = {
                        "filename": filename,
                        "path": file_path,
                        "content": text,
                        "length_chars": len(text),
                        "length_words": len(text.split())
                    }
                    documents.append(doc_info)
                    print(f"✓ Loaded {doc_info['length_chars']} chars, {doc_info['length_words']} words")
                except Exception as e:
                    print(f"⚠ Skipped {filename} (Error: {e})")

        print(f"\nTotal documents loaded: {len(documents)}")
        return documents

    def create_semantic_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from a list of documents using Chonkie.

        Args:
            documents: A list of dictionaries containing at least:
                - "content": the text of the document
                - "filename" (optional): for metadata

        Returns:
            List of chunk dictionaries with id, text, and metadata
        """
        all_chunks: List[Dict[str, Any]] = []

        for doc_idx, doc in enumerate(documents):
            text = doc.get("content", "")
            filename = doc.get("filename", f"doc_{doc_idx}")
            print(f"\nChunking document: {filename} ({len(text)} chars)")

            chunks = self.chunker.chunk(text)

            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "_id": f"{filename}_chunk_{i}",
                    "text": str(chunk),
                    "metadata": {
                        "source_file": filename,
                        "text": str(chunk),
                        "token_count": getattr(chunk, "token_count", None),
                        "chunk_index": i,
                        "doc_index": doc_idx,
                        "chunk_type": "semantic"
                    }
                }
                all_chunks.append(chunk_dict)

        print(f"\n Total chunks created: {len(all_chunks)}")
        return all_chunks

    def store_chunks_in_pinecone(
        self,
        chunks: List[Dict[str, Any]],
        namespace: str = "proof2",
        batch_size: int = 30
    ):
        """
        Store chunks in Pinecone vector database.

        Args:
            chunks: List of chunk dictionaries.
            namespace: Pinecone namespace to use.
            batch_size: Number of records per upsert batch.
        """
        try:
            if not chunks:
                print("⚠ No chunks provided for upsert.")
                return

            # Prepare upsert data (id, vector, metadata)
            upsert_data: List[Dict[str, Any]] = []
            j = 0
            for chunk in chunks:
                # Get embedding vector for the chunk
                vector = self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=chunk["text"],
                    parameters={"input_type": "query", "truncate": "END"}
                )

                # Some clients return {"data": [{"embedding": [...]}]}
                vector = vector.get("data")[0]["values"]
                upsert_data.append({
                    "id": chunk["_id"],
                    "values": vector,
                    "metadata": {
                        "text": chunk["text"],
                        **chunk.get("metadata", {})
                    }
                })
                j += 1
                print(j)

            # Batch insert
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                print(f" → Upserted batch {i // batch_size + 1} ({len(batch)} records)")

            print(f"\n Successfully stored {len(chunks)} chunks in Pinecone namespace '{namespace}'")

        except Exception as e:
            print(f"❌ Error storing chunks in Pinecone: {e}")

    def retrieve_context(self, query: str, namespace: str = "test-umsa", top_k: int = 3):
        """
        Retrieve relevant context from Pinecone based on query

        Args:
            query: Search query
            namespace: Pinecone namespace to search in
            top_k: Number of top results to return

        Returns:
            List of relevant text chunks
        """
        try:
            # Create dense query embedding
            dense_query_embedding = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=query,
                parameters={"input_type": "query", "truncate": "END"}
            )

            query_vector = dense_query_embedding.get("data")[0]["values"]

            # Search in Pinecone
            search_results = self.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )

            # Extract context from results
            contexts: List[str] = []
            for match in search_results["matches"]:
                # Original screenshots showed extracting a slice of filename; here default to text
                meta = match.get("metadata", {})
                # Prefer the text snippet if present
                if "text" in meta:
                    contexts.append(meta["text"])
                else:
                    src = meta.get("source_file", "")
                    contexts.append(src)

            return contexts
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return []

    def generate_response(self, query: str, contexts: List[str]) -> str:
        """
        Generate response using Ollama with retrieved context

        Args:
            query: User query
            contexts: Retrieved context chunks

        Returns:
            Generated response
        """
        # Prepare context string
        context_text = "\n\n".join(contexts)

        # Create enhanced prompt with context
        enhanced_prompt = f"""
Context Information:
{context_text}

Question: {query}

Based on the provided context, please answer the question. If the context doesn't contain
the answer, say "I don't know".

Answer:"""

        payload = {
            "model": self.ollama_model,
            "prompt": enhanced_prompt,
            "stream": False,
            "think": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Ollama returns {"response": "..."} in non-streaming mode
            return data.get("response", json.dumps(data))
        except Exception as e:
            return f"Error generating response: {e}"

    def process_document_pipeline(self, file_path: str, namespace: str = "proof2"):
        """
        Complete pipeline to process a document: load -> chunk -> store

        Args:
            file_path: Path to document file or folder with .md files
            namespace: Pinecone namespace to use
        """
        print(f"Processing document: {file_path}")

        # 1. Load document(s)
        print("1. Loading documents...")
        print(file_path)
        # texts = self.load_markdowns(file_path)
        process_file = ProcessFile(file_path)
        texts = process_file.load_texts()
        
        print(f"✓ Document loaded: {len(texts)} papers")

        # 2. Create semantic chunks
        print("2. Creating semantic chunks...")
        chunks = self.create_semantic_chunks(texts)
        print(f"✓ Created {len(chunks)} semantic chunks")

        # 3. Store in Pinecone
        print("3. Storing chunks in Pinecone...")
        self.store_chunks_in_pinecone(chunks, namespace)
        print("✓ Document processing complete!")

    def query_pipeline(self, query: str, namespace: str = "proof2", top_k: int = 3) -> Dict[str, Any]:
        """
        Complete query pipeline: retrieve -> generate -> return

        Args:
            query: User query
            namespace: Pinecone namespace to search in
            top_k: Number of context chunks to retrieve

        Returns:
            Dictionary with query results
        """
        print(f"Processing query: {query}")

        # 1. Retrieve context
        print("1. Retrieving relevant context...")
        contexts = self.retrieve_context(query, namespace, top_k)
        print(f"✓ Retrieved {len(contexts)} context chunks")

        # 2. Generate response
        print("2. Generating response...")
        response = self.generate_response(query, contexts)

        return {
            "query": query,
            "contexts": contexts,
            "response": response,
            "num_contexts": len(contexts)
        }


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    # Prefer environment variables instead of hardcoding secrets.
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_4WFrQ8_GkUJKEz5duWPZmdXzZMLRjoQV6Dz4J9C8vDzYhps6Cf3ESTgK1jXE6Vh8janmzf")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "proof2")
    print('++++++++')
    print(PINECONE_API_KEY)

    rag_system = OllamaRAGSystem(
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index=PINECONE_INDEX,
        ollama_model="llama2:7b-chat"  # or any model you have in Ollama
    )

    # --- Example: process a folder with markdown files ---
    document_path = "data"
    # rag_system.process_document_pipeline(document_path)

    # --- Example queries ---
    queries = [
        "Who are in on Association of Art Museum Directors?",
        # "Who is Chiara Mastroianni?", 
        # "Where was the wife of Francis I Rákóczi born?",
    ]

    # Process queries
    for query in queries:
        print("\n" + "="*50)
        result = rag_system.query_pipeline(query, namespace="proof2")

        print(f"\nQuery: {result['query']}")
        print(f"Number of contexts retrieved: {result['num_contexts']}")
        print(f"\nResponse: {result['response']}")

        # Optionally print contexts
        print(f"\nRetrieved Contexts:")
        for i, context in enumerate(result['contexts'], 1):
            print(f"{i}. {context[:200]}...")
