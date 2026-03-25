# Smart Knowledge Navigator
### An Agentic RAG System for Multi-Source Knowledge Retrieval

Built for Dell Day Ideathon @ Manipal University Jaipur


## What Is This?

Most AI chatbots either hallucinate or only search one document at a time.
This project is different — it uses three specialized AI agents that work 
together to find, evaluate, and answer questions from multiple knowledge 
sources with measurable confidence.

The system was built around a simple insight:
> "Don't just retrieve. Reason about how to retrieve."


## How It Works

### The 3-Agent Pipeline

**1. Planning Agent**
Takes the user's raw question and breaks it into 2 focused sub-questions 
using LLaMA3. This improves retrieval precision because a single vague 
query often contains multiple intents — searching with focused sub-questions 
finds better chunks from the vector database.

**2. Retrieval Agent**
Searches a FAISS vector database using semantic similarity. Instead of 
keyword matching, it converts questions into embeddings and finds the most 
semantically relevant document chunks. 

Key feature — confidence scoring:
- Converts raw FAISS distance scores into confidence percentages
- Above 50%: proceeds directly to synthesis
- 30-50%: recalibrates by asking LLaMA to rephrase the questions and searches again
- Below 30%: returns a fallback message instead of hallucinating

Confidence is reported as the highest scoring chunk's confidence — not an 
average — because the best source should determine the answer quality, 
not a diluted mean across irrelevant chunks.

**3. Synthesis Agent**
Takes retrieved chunks, builds a grounded context block with source labels, 
and asks LLaMA3 to answer using ONLY that context. The "Use ONLY the context" 
instruction in the prompt is the core anti-hallucination mechanism. Returns 
the answer with source citations and confidence score.


## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA3 via Ollama (fully local) |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Framework | LangChain |
| UI | Streamlit |
| Document Loading | LangChain DirectoryLoader + PyPDFDirectoryLoader |


## Optimisations Made During Development

**1. Confidence threshold calibration**
Initial thresholds (85%/60%) were too strict for a small prototype dataset 
with only 10-15 chunks. Recalibrated to 50%/30% to match realistic 
similarity scores from a limited vector store.

**2. Max confidence over average confidence**
Originally reported average confidence across all retrieved chunks. Changed 
to report the maximum confidence chunk's score because averaging pulls the 
score down when irrelevant chunks from unrelated files are included. The best 
source should determine the answer quality.

**3. Recalibration confidence preservation**
During recalibration, the second search sometimes returned lower quality 
chunks than the first, dragging the reported confidence down. Fixed by 
storing the original confidence and reporting max(original, recalibrated) 
so the user always sees the best confidence achieved.

**4. Multi-format document ingestion**
Expanded from TXT-only ingestion to support PDFs using PyPDFDirectoryLoader. 
This directly addresses the multi-source retrieval requirement and better 
reflects real-world enterprise knowledge bases.

**5. Query decomposition**
Added the Planning Agent after observing that single-query FAISS search 
missed relevant chunks when queries contained multiple intents. Decomposing 
into 2 sub-questions consistently improved retrieval coverage.

## Known Demerits / Limitations

**1. Small knowledge base**
The prototype uses manually created demo data with ~15 chunks. Real-world 
performance would require hundreds of documents and a larger vector store 
like Pinecone or Weaviate.

**2. No persistent chat memory**
Each query is processed independently. The system has no memory of previous 
questions in the same session — follow-up questions like "tell me more about 
that" don't work.

**3. LLaMA3 hallucination risk on edge cases**
Despite the "Use ONLY context" prompt instruction, LLaMA3 occasionally 
supplements answers with its own training knowledge when retrieved context 
is thin. A more aggressive prompt or a smaller, more instruction-tuned model 
would reduce this.

**4. Confidence scoring is approximate**
FAISS L2 distance to confidence conversion (1 - score/2.0) is a heuristic, 
not a calibrated probability. True confidence scoring would require a 
re-ranker model like cross-encoder/ms-marco-MiniLM.

**5. Single language support**
Currently only handles English queries and documents. No multilingual 
embedding model is used.

**6. No Confluence integration**
The problem statement mentions Confluence as a key source. This prototype 
simulates Confluence content via text files but does not implement actual 
Confluence API scraping.

**7. Response latency**
Running LLaMA3 locally on CPU averages 30-60 seconds per query. Production 
deployment would require GPU inference or a faster model like Mistral 7B.

---

## File Structure
```
smart-knowledge-nav/
├── data/                  # Knowledge base documents
│   ├── dell_onboarding.pdf
│   ├── dell_policies.txt
│   └── dell_tools.txt
├── faiss_index/           # Auto-generated vector store
├── ingest.py              # Document ingestion pipeline
├── agents.py              # Three agent definitions
├── config.py              # LLM parameter configuration
├── pipeline.py            # Agent orchestration
└── app.py                 # Streamlit UI
```

---

## What I Learned

This project was my first hands-on implementation of a GenAI system after 
completing an OCI Generative AI certification. The biggest insight was that 
building a RAG system is less about the LLM and more about the retrieval 
quality — garbage in, garbage out. The confidence scoring and recalibration 
logic took the most iteration to get right.
