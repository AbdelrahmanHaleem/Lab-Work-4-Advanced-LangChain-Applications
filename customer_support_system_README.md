# Enterprise Customer Support System

## Overview
The Enterprise Customer Support System is an advanced AI-powered customer support solution built using LangChain, designed to provide comprehensive support through multiple interaction channels and knowledge sources.

## Features

### 🌟 Intelligent Support Capabilities
- Multi-modal query processing
- Document-based knowledge retrieval
- Contextual conversation memory
- Product and order information lookup

### 🔍 Key Functionalities
- Order Tracking
- Product Information Retrieval
- Document-based Knowledge Base Querying
- Flexible Conversation Memory Management

## Prerequisites

### System Requirements
- Python 3.8+
- Groq API Key
- Required Python Packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/enterprise-support-system.git
cd enterprise-support-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Set up environment variables:
- Create a `.env` file in the project root
- Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Running the Application
```bash
python support_system.py
```

### Interactive Commands
- Type your query directly
- `memory:buffer` - Switch conversation memory type
- `memory:summary` - Change to summary-based memory
- `quit` - Exit the application

### Memory Types
1. `buffer`: Standard conversation history
2. `buffer_window`: Limited history window
3. `summary`: Summarized conversation context
4. `summary_buffer`: Hybrid summary and buffer approach

## Configuration

### Customizing Knowledge Base
- Place Word documents (.docx) in the knowledge base directory
- System supports dynamic document loading
- Configurable chunk size and overlap for text splitting

### Product Database
- Modify `initialize_mock_data()` to update product information
- Add or update products in the `product_database` dictionary

## Example Queries

### Order Tracking
- "What is the status of ORD-2023-1234567?"

### Product Information
- "Tell me about Enterprise Server X1"
- "Details for PRD-001"

### Document-based Queries
- Ask questions related to uploaded documentation
- System retrieves most relevant context automatically

## Advanced Configuration

### Embedding and Vector Store
- Currently uses `FakeEmbeddings` for demonstration
- Replace with actual embedding models for production

### LLM Configuration
- Modify `__init__()` to change LLM parameters
- Adjust temperature, model selection as needed

## Logging and Debugging
- Console provides detailed processing information
- Exceptions are caught and displayed gracefully

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
