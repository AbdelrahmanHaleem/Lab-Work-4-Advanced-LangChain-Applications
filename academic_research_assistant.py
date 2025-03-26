# Imports for Academic Research Assistant
from typing import List, Dict, Any, Optional
import os
import requests
import datetime
import re
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ConversationSummaryBufferMemory
import json
from dotenv import load_dotenv
import arxiv
import wikipedia
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class AcademicResearchAssistant:
    def __init__(self, api_key):
        # Initialize Groq model
        self.llm = ChatGroq(
            temperature=0.3, 
            model_name="llama3-8b-8192", 
            groq_api_key=api_key
        )
        
        # Memory strategies
        self.summary_memory = ConversationSummaryMemory(llm=self.llm, output_key="output")
        self.summary_buffer_memory = ConversationSummaryBufferMemory(
            llm=self.llm, 
            max_token_limit=100,
            output_key="output"
        )
        
        # API endpoints and tools
        self.arxiv_client = arxiv.Client()
        self.wikipedia_api = WikipediaAPIWrapper(top_k_results=3)
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=self.wikipedia_api)
        
        # Citation tracking
        self.citation_database = {}
        self.citation_count = 0
    
    def search_arxiv(self, query: str, max_results: int = 5, filter_by_relevance: bool = True) -> List[Dict[str, Any]]:
        """
        Search ArXiv for academic papers
        
        :param query: Research topic
        :param max_results: Maximum number of results
        :param filter_by_relevance: Whether to filter results by relevance score
        :return: List of research papers
        """
        try:
            # Construct a search query
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Get results
            results = list(self.arxiv_client.results(search))
            
            # Process results
            papers = []
            for result in results:
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "url": result.pdf_url,
                    "source": "arxiv",
                    "id": result.entry_id
                }
                
                # Add to citation database
                citation_id = f"arxiv-{self.citation_count}"
                self.citation_database[citation_id] = {
                    "type": "article",
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "year": result.published.year,
                    "journal": "arXiv",
                    "url": result.pdf_url,
                    "id": result.entry_id
                }
                paper["citation_id"] = citation_id
                self.citation_count += 1
                
                papers.append(paper)
            
            # Apply relevance filtering if requested
            if filter_by_relevance and papers:
                # Assess relevance with LLM
                relevance_prompt = PromptTemplate(
                    input_variables=["query", "title", "summary"],
                    template="""
                    On a scale of 1-10, how relevant is the following paper to the query: "{query}"?
                    
                    Title: {title}
                    Summary: {summary}
                    
                    Return only the numeric score between 1-10.
                    """
                )
                relevance_chain = LLMChain(llm=self.llm, prompt=relevance_prompt)
                
                filtered_papers = []
                for paper in papers:
                    try:
                        relevance_score = int(relevance_chain.run(
                            query=query,
                            title=paper["title"],
                            summary=paper["summary"]
                        ).strip())
                        
                        if relevance_score >= 6:  # Only include if relevance is 6 or higher
                            paper["relevance_score"] = relevance_score
                            filtered_papers.append(paper)
                    except:
                        # If parsing fails, include the paper anyway
                        paper["relevance_score"] = "unknown"
                        filtered_papers.append(paper)
                
                return filtered_papers
            
            return papers
            
        except Exception as e:
            print(f"Error searching ArXiv: {e}")
            return []
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for information
        
        :param query: Research topic
        :param max_results: Maximum number of results
        :return: List of Wikipedia articles
        """
        try:
            # Query Wikipedia
            search_results = self.wikipedia_tool.run(query)
            
            # Process results
            articles = []
            try:
                # Try to get full pages for the top results
                wiki_titles = wikipedia.search(query, results=max_results)
                
                for title in wiki_titles[:max_results]:
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        
                        # Create article entry
                        article = {
                            "title": page.title,
                            "summary": page.summary,
                            "url": page.url,
                            "content": page.content[:1000] + "...",  # Truncate content
                            "source": "wikipedia"
                        }
                        
                        # Add to citation database
                        citation_id = f"wiki-{self.citation_count}"
                        self.citation_database[citation_id] = {
                            "type": "webpage",
                            "title": page.title,
                            "url": page.url,
                            "accessed": datetime.datetime.now().strftime("%Y-%m-%d")
                        }
                        article["citation_id"] = citation_id
                        self.citation_count += 1
                        
                        articles.append(article)
                    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                        # Skip problematic pages
                        continue
            except:
                # If detailed page retrieval fails, use the search results
                articles = [{
                    "title": "Wikipedia Results",
                    "summary": search_results,
                    "source": "wikipedia",
                    "citation_id": f"wiki-{self.citation_count}"
                }]
                
                self.citation_database[f"wiki-{self.citation_count}"] = {
                    "type": "webpage",
                    "title": "Wikipedia Search Results",
                    "url": f"https://en.wikipedia.org/wiki/Special:Search?search={query.replace(' ', '+')}",
                    "accessed": datetime.datetime.now().strftime("%Y-%m-%d")
                }
                self.citation_count += 1
            
            return articles
            
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
            return []
    
    def filter_content_by_relevance(self, query: str, content_list: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filter content based on relevance to query
        
        :param query: Original research query
        :param content_list: List of content items
        :param threshold: Relevance threshold (0-1)
        :return: Filtered content list
        """
        if not content_list:
            return []
        
        # Create a prompt to assess relevance
        filtering_prompt = PromptTemplate(
            input_variables=["query", "title", "summary"],
            template="""
            You are an academic research assistant helping to filter content by relevance.
            
            Research Query: {query}
            
            Content Title: {title}
            Content Summary: {summary}
            
            On a scale of 0 to 1, how relevant is this content to the research query?
            Provide a single decimal number between 0 and 1.
            """
        )
        
        filtering_chain = LLMChain(llm=self.llm, prompt=filtering_prompt)
        
        filtered_content = []
        for item in content_list:
            try:
                relevance_result = filtering_chain.run(
                    query=query,
                    title=item.get("title", "No title"),
                    summary=item.get("summary", item.get("content", "No content"))
                )
                
                # Extract the numerical score
                match = re.search(r'(\d+\.\d+|\d+)', relevance_result)
                if match:
                    relevance_score = float(match.group(1))
                    
                    if relevance_score >= threshold:
                        item["relevance_score"] = relevance_score
                        filtered_content.append(item)
                else:
                    # If no score found, include by default
                    item["relevance_score"] = "unknown"
                    filtered_content.append(item)
                    
            except Exception as e:
                print(f"Error in filtering: {e}")
                # Include the item if filtering fails
                item["relevance_score"] = "error"
                filtered_content.append(item)
        
        # Sort by relevance score (if available)
        filtered_content.sort(
            key=lambda x: float(x["relevance_score"]) if isinstance(x["relevance_score"], (int, float)) else 0, 
            reverse=True
        )
        
        return filtered_content
    
    def format_citation(self, citation_id: str, style: str = "apa") -> str:
        """
        Format a citation based on its ID
        
        :param citation_id: ID of the citation in the database
        :param style: Citation style (apa, mla, chicago)
        :return: Formatted citation string
        """
        if citation_id not in self.citation_database:
            return f"[Unknown citation: {citation_id}]"
        
        citation = self.citation_database[citation_id]
        
        if style == "apa":
            if citation["type"] == "article":
                # Author(s), A. A. (Year). Title of article. Journal Name, Volume(Issue), pages. URL
                authors = ", ".join(citation.get("authors", ["Unknown"]))
                return f"{authors} ({citation.get('year', 'n.d.')}). {citation.get('title', 'Untitled')}. {citation.get('journal', 'Journal')}. {citation.get('url', '')}"
            
            elif citation["type"] == "webpage":
                # Author(s). (Year, Month Day). Title. Site Name. URL
                return f"{citation.get('title', 'Untitled')}. (Accessed: {citation.get('accessed', 'n.d.')}). {citation.get('url', '')}"
            
        elif style == "mla":
            if citation["type"] == "article":
                # Author(s). "Title of Article." Journal Name, vol. Volume, no. Issue, Year, pp. Pages.
                authors = ", ".join(citation.get("authors", ["Unknown"]))
                return f"{authors}. \"{citation.get('title', 'Untitled')}\" {citation.get('journal', 'Journal')}, {citation.get('year', 'n.d.')}. {citation.get('url', '')}"
            
            elif citation["type"] == "webpage":
                # Author. "Title." Site Name, Publisher, Date published, URL.
                return f"\"{citation.get('title', 'Untitled')}\" Web. {citation.get('accessed', 'n.d.')}. {citation.get('url', '')}"
        
        # Default format if style not recognized
        return f"{citation.get('title', 'Untitled')} ({citation_id}). {citation.get('url', '')}"
    
    def generate_citations(self, citation_ids: List[str], style: str = "apa") -> str:
        """
        Generate a bibliography from citation IDs
        
        :param citation_ids: List of citation IDs to include
        :param style: Citation style
        :return: Formatted bibliography
        """
        citations = []
        for cid in citation_ids:
            citations.append(self.format_citation(cid, style))
        
        return "\n\n".join(citations)
    
    def generate_research_report(self, query: str, output_format: str = 'json', include_citations: bool = True, filter_results: bool = True) -> str:
        """
        Generate a comprehensive research report by synthesizing information from multiple sources
        
        :param query: Research topic
        :param output_format: Report output format (json, markdown, text)
        :param include_citations: Whether to include source citations
        :param filter_results: Whether to filter results by relevance
        :return: Formatted research report
        """
        # Reset citation database for this query
        self.citation_database = {}
        self.citation_count = 0
        
        # Search multiple sources
        arxiv_results = self.search_arxiv(query, max_results=5, filter_by_relevance=filter_results)
        wikipedia_results = self.search_wikipedia(query, max_results=3)
        
        # Combine and filter results if requested
        all_sources = arxiv_results + wikipedia_results
        if filter_results:
            filtered_sources = self.filter_content_by_relevance(query, all_sources)
        else:
            filtered_sources = all_sources
        
        # Prepare sources for the synthesis
        sources_text = ""
        used_citation_ids = []
        
        for idx, source in enumerate(filtered_sources, 1):
            citation_id = source.get("citation_id", f"source-{idx}")
            used_citation_ids.append(citation_id)
            
            sources_text += f"\n\nSOURCE {idx} [ID: {citation_id}]:\n"
            sources_text += f"Title: {source.get('title', 'Untitled')}\n"
            sources_text += f"Source Type: {source.get('source', 'unknown')}\n"
            
            if "authors" in source:
                sources_text += f"Authors: {', '.join(source['authors'])}\n"
            
            sources_text += f"Content: {source.get('summary', source.get('content', 'No content available'))}\n"
        
        # Use a sequential chain for synthesis
        # Step 1: Initial synthesis of information
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "sources"],
            template="""
            You are an academic research assistant conducting research on: {query}
            
            Please synthesize the following sources into a cohesive initial research analysis:
            
            {sources}
            
            Focus on finding patterns, agreements, disagreements, and key findings across the sources.
            Tag important information with the source ID (e.g., [ID: arxiv-1]).
            """
        )
        
        # Step 2: Identify gaps and questions
        gap_analysis_prompt = PromptTemplate(
            input_variables=["query", "initial_synthesis"],
            template="""
            Based on the initial synthesis of research on {query}:
            
            {initial_synthesis}
            
            Identify:
            1. Key gaps in the current research
            2. Contradictions or disagreements among sources
            3. The most important open questions
            4. Areas that need further investigation
            
            Frame your analysis in an academic context.
            """
        )
        
        # Step 3: Final report generation
        report_prompt = PromptTemplate(
            input_variables=["query", "initial_synthesis", "gap_analysis"],
            template="""
            You are preparing a formal academic research report on: {query}
            
            Based on the synthesis of available sources:
            
            {initial_synthesis}
            
            And the identified research gaps:
            
            {gap_analysis}
            
            Generate a comprehensive research report with the following sections:
            1. Introduction and Background
            2. Key Findings
            3. Analysis of Current Research
            4. Research Gaps and Future Directions
            5. Conclusion
            
            Make sure to reference sources using their IDs (e.g., [ID: arxiv-1]).
            Write in a formal academic style suitable for a research paper.
            """
        )
        
        # Configure chains
        synthesis_chain = LLMChain(llm=self.llm, prompt=synthesis_prompt, output_key="initial_synthesis")
        gap_chain = LLMChain(llm=self.llm, prompt=gap_analysis_prompt, output_key="gap_analysis")
        report_chain = LLMChain(llm=self.llm, prompt=report_prompt, output_key="report")
        
        # Create sequential chain
        research_chain = SequentialChain(
            chains=[synthesis_chain, gap_chain, report_chain],
            input_variables=["query", "sources"],
            output_variables=["report", "initial_synthesis", "gap_analysis"],
            verbose=False
        )
        
        try:
            # Run the chain
            result = research_chain({"query": query, "sources": sources_text})
            report_content = result["report"]
            
            # Generate citations if requested
            citations_text = ""
            if include_citations and used_citation_ids:
                citations_text = "\n\n## References\n\n" + self.generate_citations(used_citation_ids)
            
            # Prepare report
            timestamp = datetime.datetime.now().isoformat()
            
            if output_format == 'json':
                report = {
                    "title": f"Research Report: {query}",
                    "content": report_content,
                    "sources": [
                        {
                            "id": source.get("citation_id", f"source-{i}"),
                            "title": source.get("title", "Untitled"),
                            "type": source.get("source", "unknown"),
                            "relevance": source.get("relevance_score", "unknown")
                        } for i, source in enumerate(filtered_sources, 1)
                    ],
                    "citation_count": len(used_citation_ids),
                    "timestamp": timestamp
                }
                
                if include_citations:
                    report["citations"] = self.generate_citations(used_citation_ids)
                
                return json.dumps(report, indent=2)
                
            elif output_format == 'markdown':
                md_report = f"# Research Report: {query}\n\n"
                md_report += report_content
                
                if include_citations:
                    md_report += f"\n\n## References\n\n"
                    for cid in used_citation_ids:
                        md_report += f"- {self.format_citation(cid)}\n"
                
                md_report += f"\n\n*Generated at: {timestamp}*"
                
                return md_report
                
            else:  # text format
                text_report = f"RESEARCH REPORT: {query}\n"
                text_report += "=" * 50 + "\n\n"
                text_report += report_content
                
                if include_citations:
                    text_report += "\n\nREFERENCES:\n"
                    text_report += "-" * 50 + "\n"
                    for cid in used_citation_ids:
                        text_report += f"{self.format_citation(cid)}\n\n"
                
                text_report += f"\nGenerated at: {timestamp}"
                
                return text_report
                
        except Exception as e:
            error_message = f"Error generating research report: {e}"
            
            if output_format == 'json':
                return json.dumps({
                    "error": error_message,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                return f"ERROR: {error_message}"

    def synthesize_research(self, topic: str, sources: List[Dict[str, Any]]) -> str:
        """
        Synthesize research from multiple sources
        
        :param topic: Research topic
        :param sources: List of research sources
        :return: Synthesized research summary
        """
        try:
            # Prepare sources for synthesis
            sources_text = ""
            for idx, source in enumerate(sources, 1):
                citation_id = source.get("citation_id", f"source-{idx}")
                sources_text += f"\n\nSOURCE {idx} [ID: {citation_id}]:\n"
                sources_text += f"Title: {source.get('title', 'Untitled')}\n"
                sources_text += f"Source Type: {source.get('source', 'unknown')}\n"
                
                if "authors" in source:
                    sources_text += f"Authors: {', '.join(source['authors'])}\n"
                
                sources_text += f"Content: {source.get('summary', source.get('content', 'No content available'))}\n"
            
            # Create synthesis prompt
            synthesis_prompt = PromptTemplate(
                input_variables=["query", "sources"],
                template="""
                You are an academic research assistant conducting research on: {query}
                
                Please synthesize the following sources into a cohesive research analysis:
                
                {sources}
                
                Focus on finding patterns, agreements, disagreements, and key findings across the sources.
                Tag important information with the source ID (e.g., [ID: arxiv-1]).
                Organize your synthesis into sections like:
                1. Introduction
                2. Key Findings
                3. Current Research Gaps
                4. Conclusions
                """
            )
            
            # Create synthesis chain
            synthesis_chain = LLMChain(llm=self.llm, prompt=synthesis_prompt)
            
            # Run synthesis
            result = synthesis_chain.run(query=topic, sources=sources_text)
            
            return result
            
        except Exception as e:
            print(f"Error synthesizing research: {e}")
            return f"Error synthesizing research: {e}"
            
    def answer_research_question(self, question: str, research_topic: str, sources: List[Dict[str, Any]]) -> str:
        """
        Answer a research question based on collected sources
        
        :param question: The research question to answer
        :param research_topic: The main research topic
        :param sources: List of research sources
        :return: Answer to the research question
        """
        try:
            # Prepare sources for analysis
            sources_text = ""
            for idx, source in enumerate(sources, 1):
                citation_id = source.get("citation_id", f"source-{idx}")
                sources_text += f"\n\nSOURCE {idx} [ID: {citation_id}]:\n"
                sources_text += f"Title: {source.get('title', 'Untitled')}\n"
                
                if "authors" in source:
                    sources_text += f"Authors: {', '.join(source['authors'])}\n"
                
                sources_text += f"Content: {source.get('summary', source.get('content', 'No content available'))}\n"
            
            # Create question prompt
            question_prompt = PromptTemplate(
                input_variables=["research_topic", "question", "sources"],
                template="""
                You are an academic research assistant helping with research on: {research_topic}
                
                Please answer the following question based only on the provided sources:
                
                QUESTION: {question}
                
                SOURCES:
                {sources}
                
                Format your answer:
                1. Provide a direct answer to the question
                2. Include supporting evidence from the sources
                3. Cite sources using their ID (e.g., [ID: arxiv-1])
                4. Note any limitations or areas where the sources don't provide sufficient information
                """
            )
            
            # Create question chain
            question_chain = LLMChain(llm=self.llm, prompt=question_prompt)
            
            # Run question answering
            result = question_chain.run(
                research_topic=research_topic,
                question=question,
                sources=sources_text
            )
            
            return result
            
        except Exception as e:
            print(f"Error answering research question: {e}")
            return f"I couldn't answer that question due to an error: {e}"
    
def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        return
    
    research_assistant = AcademicResearchAssistant(api_key)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*50)
    print(" Academic Research Assistant ".center(50, "="))
    print("="*50)
    print("\nYour AI-powered research companion\n")
    print("Available commands:")
    print("  - 'search <topic>': Search for a research topic")
    print("  - 'summarize': Generate a summary of collected research")
    print("  - 'citations': Show all citations for your research")
    print("  - 'save <filename>': Save your research to a file")
    print("  - 'help': Show this help menu")
    print("  - 'quit': Exit the application")
    print("="*50 + "\n")
    
    current_research_topic = None
    collected_sources = []
    research_summary = None
    
    while True:
        user_input = input("\n🔍 Research query: ")
        
        if user_input.lower() == 'quit':
            print("\nThank you for using the Academic Research Assistant. Goodbye!")
            break
            
        elif user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("  - 'search <topic>': Search for a research topic")
            print("  - 'summarize': Generate a summary of collected research")
            print("  - 'citations': Show all citations for your research")
            print("  - 'save <filename>': Save your research to a file")
            print("  - 'help': Show this help menu")
            print("  - 'quit': Exit the application")
            
        elif user_input.lower().startswith('search '):
            topic = user_input[7:].strip()
            if not topic:
                print("Please specify a research topic after 'search'.")
                continue
                
            print(f"\n📚 Researching: {topic}")
            current_research_topic = topic
            
            print("Searching ArXiv for academic papers...")
            arxiv_results = research_assistant.search_arxiv(topic, max_results=3)
            
            print("Searching Wikipedia for general information...")
            wiki_results = research_assistant.search_wikipedia(topic, max_results=2)
            
            collected_sources = arxiv_results + wiki_results
            
            print(f"\n✅ Found {len(collected_sources)} sources:")
            for i, source in enumerate(collected_sources, 1):
                print(f"  {i}. {source['title']} ({source['source']})")
            
            print("\nUse 'summarize' to generate a research summary.")
            
        elif user_input.lower() == 'summarize':
            if not current_research_topic or not collected_sources:
                print("Please search for a topic first before generating a summary.")
                continue
                
            print(f"\n📝 Generating research summary for '{current_research_topic}'...")
            
            # Generate summary
            try:
                research_summary = research_assistant.synthesize_research(
                    topic=current_research_topic,
                    sources=collected_sources
                )
                
                # Display summary
                print("\n===== RESEARCH SUMMARY =====\n")
                print(research_summary)
                print("\n=============================")
            except Exception as e:
                print(f"Error generating summary: {e}")
            
        elif user_input.lower() == 'citations':
            if not research_assistant.citation_database:
                print("No citations available. Please search for a topic first.")
                continue
                
            print("\n📚 Citations:\n")
            for citation_id, citation in research_assistant.citation_database.items():
                if citation["type"] == "article":
                    print(f"[{citation_id}] {', '.join(citation['authors'][:3])}" + 
                          (", et al." if len(citation['authors']) > 3 else "") + 
                          f" ({citation['year']}). {citation['title']}. {citation['journal']}. {citation['url']}")
                elif citation["type"] == "webpage":
                    print(f"[{citation_id}] {citation['title']}. Retrieved on {citation['accessed']} from {citation['url']}")
            
        elif user_input.lower().startswith('save '):
            filename = user_input[5:].strip()
            if not filename:
                print("Please specify a filename after 'save'.")
                continue
                
            if not research_summary:
                print("No research summary to save. Please use 'summarize' first.")
                continue
                
            if not filename.endswith('.txt'):
                filename += '.txt'
                
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"RESEARCH SUMMARY: {current_research_topic.upper()}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write(research_summary + "\n\n")
                    
                    f.write("REFERENCES\n")
                    f.write("-" * 80 + "\n\n")
                    for citation_id, citation in research_assistant.citation_database.items():
                        if citation["type"] == "article":
                            f.write(f"[{citation_id}] {', '.join(citation['authors'][:3])}" + 
                                  (", et al." if len(citation['authors']) > 3 else "") + 
                                  f" ({citation['year']}). {citation['title']}. {citation['journal']}. {citation['url']}\n")
                        elif citation["type"] == "webpage":
                            f.write(f"[{citation_id}] {citation['title']}. Retrieved on {citation['accessed']} from {citation['url']}\n")
                
                print(f"\n✅ Research summary saved to '{filename}'.")
            except Exception as e:
                print(f"Error saving file: {e}")
            
        else:
            # Handle general research questions
            if current_research_topic:
                print(f"\n🤔 Analyzing your question about '{current_research_topic}'...")
                
                try:
                    response = research_assistant.answer_research_question(
                        question=user_input,
                        research_topic=current_research_topic,
                        sources=collected_sources
                    )
                    
                    print(f"\n🔬 Research Answer:")
                    print(response)
                except Exception as e:
                    print(f"Error answering question: {e}")
            else:
                print("\nPlease search for a topic first using 'search <topic>' or use one of the available commands.")
                print("Type 'help' to see the list of commands.")

if __name__ == "__main__":
    main()