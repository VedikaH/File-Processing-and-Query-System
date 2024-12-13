import os
import faiss
import torch
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

class RAGQueryHandler:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # FAISS index
        self.faiss_index = None
        self.embeddings_list = []
        self.passages = []

    def create_embeddings_index(self, extracted_content: Dict[str, Any]):
        """
        Create FAISS index for document embeddings
        
        Args:
            extracted_content (Dict): Content extracted from PDF
        """
        # Prepare passages
        self.passages = []
        
        # Extract text passages
        for text_entry in extracted_content.get('text', []):
            # Split text into smaller chunks for better embedding
            chunks = self._split_text(text_entry['content'])
            for chunk in chunks:
                self.passages.append({
                    'text': chunk,
                    'page_number': text_entry['page_number']
                })
        
        # Extract table passages
        for table_entry in extracted_content.get('tables', []):
            table_text = self._convert_table_to_text(table_entry['data'])
            self.passages.append({
                'text': table_text,
                'page_number': table_entry['page_number']
            })
            
        print("The passages are:", self.passages)
              
        # Create embeddings
        embeddings = self.embedding_model.encode([p['text'] for p in self.passages])
        self.embeddings_list = embeddings
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(embeddings).astype('float32'))

    def _split_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks
        
        Args:
            text (str): Input text
            max_chunk_size (int): Maximum chunk size
        
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(word)
            current_length += len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print("The chunks are:",chunks)
        return chunks

    def _convert_table_to_text(self, table: List[List[str]]) -> str:
        """
        Convert table to textual representation
        
        Args:
            table (List[List[str]]): Table data
        
        Returns:
            str: Textual representation of table
        """
        table_text = []
        for row in table:
            table_text.append(' | '.join(str(cell) for cell in row if cell is not None))
        return '\n'.join(table_text)

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant context for a query
        
        Args:
            query (str): User query
            top_k (int): Number of top relevant passages to retrieve
        
        Returns:
            List of relevant passages
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not initialized. Call create_embeddings_index first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]).astype('float32'), 
            top_k
        )
        
        # Retrieve relevant passages
        relevant_passages = []
        for idx in indices[0]:
            relevant_passages.append({
                'text': self.passages[idx]['text'],
                'page_number': self.passages[idx]['page_number']
            })
        
        return relevant_passages

    def query_document(self, extracted_content: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Query the document using Groq with RAG approach
        
        Args:
            extracted_content (Dict): Content extracted from PDF
            query (str): User's query
        
        Returns:
            Dict: Structured query response
        """
        try:
            # Create FAISS index if not already created
            if self.faiss_index is None:
                self.create_embeddings_index(extracted_content)
            
            # Retrieve relevant context
            relevant_contexts = self.retrieve_relevant_context(query)
            
            # Construct context-aware prompt
            context = "\n\n".join([f"Passage from page {ctx['page_number']}: {ctx['text']}" for ctx in relevant_contexts])
            
            # Query Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise and accurate answers based on the given context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuery: {query}\n\nProvide a comprehensive answer based on the context."
                    }
                ],
                model="gemma2-9b-it"
            )
            
            response_text = chat_completion.choices[0].message.content
            
            return {
                "query": query,
                "response": {
                    "summary": response_text
                },
                "source_pages": list(set(ctx['page_number'] for ctx in relevant_contexts))
            }
        
        except Exception as e:
            return {
                "query": query,
                "response": {"error": str(e)},
                "source_pages": []
            }

   