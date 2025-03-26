import os
from typing import Dict, List, Any, Optional, Union
import json
import re
import random
import getpass
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema import SystemMessage
from dotenv import load_dotenv
import docx
from io import BytesIO

class EnterpriseCustomerSupportSystem:
    def __init__(self, api_key):
        self.llm = ChatGroq(
            temperature=0.2, 
            model_name="llama3-8b-8192", 
            groq_api_key=api_key
        )
        
        # Initialize memory
        self.memory_options = {
            "buffer": ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            ),
            "buffer_window": ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=5,
                return_messages=True
            ),
            "summary": ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            ),
            "summary_buffer": ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                max_token_limit=100,
                return_messages=True
            )
        }
        
        self.active_memory = "buffer"
        self.kb_directories = ["kb/hardware", "kb/software", "kb/services"]
        self.document_knowledge_base = None
        self.vectorstore = None
        
        # Create order tracking and product database
        self.initialize_mock_data()
        
        # Default prompt setup for general support
        self.general_prompt = ChatPromptTemplate(
            [
                SystemMessage(content="""You are an Enterprise Customer Support AI assistant. 
                Your role is to provide helpful, accurate, and concise responses to customer inquiries.
                Base your answers only on the provided context and information. 
                If you don't know the answer, say so rather than making something up."""),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{text}")
            ]
        )
        
        # Document-based support prompt - simplified for direct RAG
        self.document_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an Enterprise Customer Support AI assistant.
                Answer the question based only on the provided document context.
                Be concise and direct in your response, focusing only on answering the specific question.
                If you don't know the answer from the given context, just say you don't have that information."""),
            HumanMessagePromptTemplate.from_template("""Context information is below:
                ---------------------
                {context}
                ---------------------
                Given the context information and not prior knowledge, answer the question: {question}""")
        ])
        
        # Set up the default chain
        self.general_chain = LLMChain(
            llm=self.llm,
            prompt=self.general_prompt,
            memory=self.memory_options[self.active_memory],
            verbose=False
        )
        
    def initialize_mock_data(self):
        """
        Initialize mock data for order tracking and product information
        """
        # Product information database
        self.product_database = {
            "PRD-001": {
                "name": "Enterprise Server X1",
                "category": "Hardware",
                "price": 2999.99,
                "warranty": "3 years",
                "details": {
                    "cpu": "Intel Xeon 8-core",
                    "ram": "64GB ECC",
                    "storage": "2TB NVMe SSD",
                    "os": "Linux/Windows Server compatible"
                }
            },
            "PRD-002": {
                "name": "Business Analytics Suite",
                "category": "Software",
                "price": 199.99,
                "warranty": "1 year support",
                "details": {
                    "version": "2023.2",
                    "license": "Annual subscription",
                    "users": "Up to 50 concurrent users",
                    "features": "Reporting, dashboards, predictive analytics"
                }
            },
            "PRD-003": {
                "name": "Network Security Package",
                "category": "Services",
                "price": 4999.99,
                "warranty": "1 year",
                "details": {
                    "coverage": "24/7 monitoring",
                    "response_time": "15 minutes",
                    "includes": "Firewall, IDS, threat analysis, monthly reports"
                }
            }
        }
        
        # Order database for tracking
        self.order_database = {
            "ORD-2023-1234567": {
                "status": "Shipped",
                "date": "2023-10-15",
                "products": ["Enterprise Server X1", "Network Security Package"],
                "shipping_method": "Express",
                "estimated_delivery": "2023-10-20"
            },
            "ORD-2023-7654321": {
                "status": "Processing",
                "date": "2023-11-05",
                "products": ["Business Analytics Suite"],
                "shipping_method": "Standard",
                "estimated_delivery": "2023-11-12"
            },
            "ORD-2023-9876543": {
                "status": "Delivered",
                "date": "2023-09-20",
                "products": ["Network Security Package"],
                "shipping_method": "Express",
                "delivered_date": "2023-09-22"
            }
        }
    
    def initialize_document_knowledge_base(self, docx_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """
        Initialize the knowledge base from a Word document
        
        Args:
            docx_path: Path to the Word document
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        try:
            # Check if the file exists
            if not os.path.exists(docx_path):
                print(f"Document not found: {docx_path}")
                return
            
            print(f"Loading document: {docx_path}")
            
            # Extract text from the document
            loader = Docx2txtLoader(docx_path)
            documents = loader.load()
            
            print(f"Document loaded successfully with {len(documents)} pages")
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            
            print(f"Created {len(chunks)} chunks from the document")
            
            # Create a vector store
            embeddings = FakeEmbeddings(size=1536)  # Using FakeEmbeddings for demonstration
            self.document_knowledge_base = FAISS.from_documents(chunks, embeddings)
            
            print(f"Vector store created successfully")
            print(f"Document knowledge base is ready for querying")
            
        except Exception as e:
            print(f"Error initializing document knowledge base: {e}")
            import traceback
            traceback.print_exc()
    
    def load_documents(self, document_paths):
        """
        Load and process multiple documents for the knowledge base
        """
        docs = []
        for path in document_paths:
            try:
                if path.endswith('.pdf'):
                    loader = DirectoryLoader(os.path.dirname(path), glob=os.path.basename(path))
                elif path.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                else:
                    loader = TextLoader(path)
                
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if docs:
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(
                documents=split_docs, 
                embedding=FakeEmbeddings(size=1536)  # Using FakeEmbeddings for demonstration
            )
    
    def track_order(self, query):
        """
        Track order status based on order ID
        
        Args:
            query: User query containing order ID
            
        Returns:
            Dict with order tracking information
        """
        # Extract order ID using regex pattern
        order_pattern = r'ORD-\d{4}-\d{7}'
        order_ids = re.findall(order_pattern, query)
        
        if not order_ids:
            return {
                "found": False,
                "message": "No order ID found in your query. Please provide your order ID in the format ORD-YYYY-NNNNNNN."
            }
        
        order_id = order_ids[0]  # Use the first found order ID
        
        # Look up the order in the database
        if order_id in self.order_database:
            return {
                "found": True,
                "order_id": order_id,
                **self.order_database[order_id]
            }
        else:
            return {
                "found": False,
                "message": f"Order {order_id} not found in our system. Please verify the order ID and try again."
            }
    
    def lookup_product_info(self, query):
        """
        Look up product information based on product ID or name
        
        Args:
            query: User query containing product ID or name
            
        Returns:
            Dict with product information
        """
        # Check for direct product ID match
        for product_id, info in self.product_database.items():
            if product_id in query:
                return {
                    "found": True,
                    "product_id": product_id,
                    **info
                }
        
        # Check for product name match
        for product_id, info in self.product_database.items():
            if info["name"].lower() in query.lower():
                return {
                    "found": True,
                    "product_id": product_id,
                    **info
                }
        
        # No product found
        return {
            "found": False,
            "message": "No specific product identified in your query. Please provide a product ID or name."
        }
    
    def query(self, user_query: str, memory_type: str = None) -> Dict[str, Any]:
        """
        Process user query and generate response
        
        Args:
            user_query: User's query text
            memory_type: Type of memory to use (buffer, buffer_window, summary, summary_buffer)
            
        Returns:
            Dict containing query, response, and source information
        """
        try:
            # Set memory type if provided
            if memory_type and memory_type in self.memory_options:
                self.active_memory = memory_type
                self.general_chain.memory = self.memory_options[self.active_memory]
            
            # Check for order tracking requests
            if self.is_order_tracking_query(user_query):
                tracking_info = self.track_order(user_query)
                if tracking_info["found"]:
                    return {
                        "query": user_query,
                        "support_type": "order_tracking",
                        "response": f"Order {tracking_info['order_id']} status: {tracking_info['status']}. Ordered on {tracking_info['date']}. Shipping method: {tracking_info['shipping_method']}.",
                        "structured_data": tracking_info,
                        "sources": []
                    }
                else:
                    return {
                        "query": user_query,
                        "support_type": "order_tracking",
                        "response": tracking_info["message"],
                        "sources": []
                    }
            
            # Check for product lookup requests
            if self.is_product_lookup_query(user_query):
                product_info = self.lookup_product_info(user_query)
                if product_info["found"]:
                    details = []
                    for key, value in product_info["details"].items():
                        details.append(f"{key.replace('_', ' ').title()}: {value}")
                    
                    return {
                        "query": user_query,
                        "support_type": "product_info",
                        "response": f"Product: {product_info['name']} ({product_info['product_id']})\nCategory: {product_info['category']}\nPrice: ${product_info['price']}\n\nDetails:\n" + "\n".join(details),
                        "structured_data": product_info,
                        "sources": []
                    }
                else:
                    return {
                        "query": user_query,
                        "support_type": "general",
                        "response": product_info["message"],
                        "sources": []
                    }
            
            # Handle document-based queries if document knowledge base is available
            if self.document_knowledge_base:
                try:
                    # Get relevant context from the document knowledge base
                    docs = self.document_knowledge_base.similarity_search(user_query, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Use a direct approach without memory for document queries
                    # This avoids the issue with multiple output keys
                    from langchain.chains import LLMChain
                    
                    # Create a new chain without memory for each document query
                    document_chain = LLMChain(
                        llm=self.llm,
                        prompt=self.document_prompt,
                        verbose=False  # Set to False to prevent printing the prompt
                    )
                    
                    # Run the document chain with the correct input keys
                    document_response = document_chain.invoke({
                        "question": user_query,
                        "context": context
                    })
                    
                    # Extract source information for response
                    source_info = []
                    for doc in docs:
                        source_info.append({
                            "source": "document",
                            "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                            "metadata": doc.metadata
                        })
                    
                    # Extract the response text, handling different return types
                    response_text = ""
                    if isinstance(document_response, dict) and "text" in document_response:
                        response_text = document_response["text"]
                    elif isinstance(document_response, dict) and "output_text" in document_response:
                        response_text = document_response["output_text"]
                    elif hasattr(document_response, "content"):
                        response_text = document_response.content
                    else:
                        response_text = str(document_response)
                    
                    # For memory continuity, add this exchange to the active memory
                    self.memory_options[self.active_memory].save_context(
                        {"text": user_query}, 
                        {"output": response_text}
                    )
                    
                    return {
                        "query": user_query,
                        "support_type": "document_knowledge_base",
                        "response": response_text.strip(),
                        "sources": source_info,
                        "memory_type": self.active_memory
                    }
                except Exception as e:
                    print(f"Error processing document query: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Use the general chain for other queries
            general_response = self.general_chain.run(text=user_query)
            
            return {
                "query": user_query,
                "support_type": "general",
                "response": general_response.strip(),
                "sources": [],
                "memory_type": self.active_memory
            }
            
        except Exception as e:
            # Handle errors gracefully
            error_response = {
                "query": user_query,
                "error": str(e),
                "support_type": "error",
                "response": "Sorry, I encountered an error while processing your request. Please try again."
            }
            return error_response
    
    def is_order_tracking_query(self, query):
        order_pattern = r'ORD-\d{4}-\d{7}'
        return bool(re.findall(order_pattern, query))

    def is_product_lookup_query(self, query):
        for product_id in self.product_database:
            if product_id in query:
                return True
        for product_name in [info["name"] for info in self.product_database.values()]:
            if product_name.lower() in query.lower():
                return True
        return False

    def get_support_response(self, user_query):
        """
        Process a support query and return a formatted response
        
        :param user_query: The user's support query
        :return: Formatted response dictionary
        """
        try:
            # Process the query and get the response
            response = self.query(user_query)
            return response
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": user_query,
                "support_type": "error",
                "response": f"Sorry, I encountered an error while processing your query: {e}",
                "sources": [],
                "memory_type": self.active_memory
            }

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        return
    
    # Create a support system instance
    support_system = EnterpriseCustomerSupportSystem(api_key)
    
    # Interactive session
    print("\n========================================")
    print(" Enterprise Customer Support System")
    print("========================================")
    print("Enter 'quit' to exit")
    print("Enter 'memory:[type]' to change memory type")
    print("Available memory types: buffer, buffer_window, summary, summary_buffer")
    print("========================================\n")
    
    # Option to load a document
    print("To use document-based knowledge, provide a Word document (.docx) file.")
    doc_path = input("Enter path to Word document for knowledge base (or press Enter to skip): ")
    
    if doc_path:
        if os.path.exists(doc_path):
            print("\nInitializing document knowledge base...")
            support_system.initialize_document_knowledge_base(doc_path)
            print("Document knowledge base initialized successfully!")
        else:
            print(f"\nDocument not found: {doc_path}")
            # Try with a default path if user input isn't found
            default_path = "Enterprise Customer Support Documentation.docx"
            if os.path.exists(default_path):
                print(f"Using default document: {default_path}")
                support_system.initialize_document_knowledge_base(default_path)
                print("Default document knowledge base initialized successfully!")
    
    print("\n========================================")
    print(" Ready to answer your questions!")
    print("========================================\n")
    
    while True:
        # Get user input
        user_input = input("\n How can I help you today? ")
        
        # Check for quit command
        if user_input.lower() == 'quit':
            print("\nThank you for using the Enterprise Customer Support System. Goodbye!")
            break
        
        # Check for memory type change
        if user_input.startswith('memory:'):
            memory_type = user_input.split(':')[1].strip()
            if memory_type in support_system.memory_options:
                support_system.active_memory = memory_type
                support_system.general_chain.memory = support_system.memory_options[memory_type]
                print(f"\n Switched to {memory_type} memory")
            else:
                print(f"\n Unknown memory type: {memory_type}")
                print(f"Available options: {', '.join(support_system.memory_options.keys())}")
            continue
        
        # Process query
        print("\n Processing your query...")
        response = support_system.get_support_response(user_input)
        
        # Print response
        if "error" in response:
            print(f"\n Error: {response['error']}")
        else:
            print(f"\n {response['response']}")
            
            # Print source information (optional)
            if "sources" in response and response["sources"] and False:  # Disabled for cleaner output
                print("\nSources:")
                for idx, source in enumerate(response["sources"], 1):
                    print(f"{idx}. {source.get('source', 'Unknown')}")
        
if __name__ == "__main__":
    main()