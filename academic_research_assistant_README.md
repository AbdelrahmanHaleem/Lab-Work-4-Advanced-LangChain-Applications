# **Academic Research Assistant** 🎓🔍  

A **LangChain-powered** AI research assistant that helps collect, synthesize, and organize academic information from multiple sources.  

🚀 **Features:**  
- **Multi-source research** (ArXiv, Wikipedia)  
- **Smart synthesis** using sequential LLM chains  
- **Structured report generation** (JSON, Markdown, Text)  
- **Citation management** (APA, MLA, Chicago)  
- **Relevance filtering** (LLM-powered scoring)  
- **Interactive CLI** for easy usage  

---

## **Installation**  

### **Prerequisites**  
- Python 3.9+  
- [Groq API Key](https://console.groq.com/) (Free tier available)  

### **Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/AbdelrahmanHaleem/Lab-Work-4-Advanced-LangChain-Applications
   cd academic-research-assistant
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:  
   - Create a `.env` file:  
     ```bash
     touch .env
     ```
   - Add your **Groq API Key**:  
     ```
     GROQ_API_KEY=your_api_key_here
     ```

---

## **Usage**  

### **Run the CLI**  
```bash
python research_assistant.py
```

### **Available Commands**  
| Command | Description |
|---------|-------------|
| `search <topic>` | Search for research papers & Wikipedia articles |
| `summarize` | Generate a synthesized research summary |
| `citations` | View all citations in APA/MLA format |
| `save <filename>` | Save report in JSON/Markdown/Text |
| `quit` | Exit the program |

### **Example Workflow**  
1. **Search for a topic:**  
   ```
   search quantum computing
   ```
2. **Generate a summary:**  
   ```
   summarize
   ```
3. **Save the report:**  
   ```
   save quantum_research.md
   ```

---

## **Features in Detail**  

### **1. Multi-Source Research**  
- **ArXiv API** → Latest academic papers  
- **Wikipedia** → General knowledge & definitions  

### **2. Smart Synthesis**  
- **SequentialChain** for:  
  - Initial synthesis  
  - Gap analysis  
  - Final report generation  

### **3. Report Formats**  
- **JSON** (structured metadata)  
- **Markdown** (readable formatting)  
- **Plain Text** (simple output)  

### **4. Citation Management**  
- Auto-generates citations in:  
  - **APA**  
  - **MLA**  
  - **Chicago**  

### **5. Relevance Filtering**  
- **LLM-powered scoring** (1-10 scale)  
- Filters out irrelevant sources  

---

## **Future Enhancements**  
- [ ] Add **Google Scholar/PubMed** integration  
- [ ] Support **PDF/URL ingestion**  
- [ ] **Web-based UI** (Streamlit/Gradio)  

