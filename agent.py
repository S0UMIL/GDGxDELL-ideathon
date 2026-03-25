from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from config import LLM_MODEL, TEMPERATURE, NUM_PREDICT, TOP_P, TOP_K


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_vectorstore():#extracting all the data from faiss vectors
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT,
        top_p=TOP_P,
        top_k=TOP_K
    )

def planning_agent(query, llm):# in this we basically tell llama bhai to split the query in to sub-questions
    print(f"\n[Planning Agent] Original query: {query}")
    prompt = f"""You are a query planning assistant.
Given the user's question, break it down into 2 clear and specific sub-questions
that together would help answer the original question completely.

Original question: {query}

Return exactly 2 sub-questions, one per line, no numbering, no extra text."""
    response = llm.invoke(prompt)#ask llama to generate
    sub_questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
    sub_questions = sub_questions[:2]#making sure only first 2 are going through as llama can give 3-4 sub questions
    print(f"[Planning Agent] Sub-questions: {sub_questions}")
    return sub_questions

def retrieval_agent(sub_questions, vectorstore, llm):#this agent is for our overall performance of our RAG system , tells how the system is performing confidence score etc.
    print(f"\n[Retrieval Agent] Searching for {len(sub_questions)} sub-questions...")
    all_chunks = []
    seen_contents = set()
    overall_confidence = 0
    total_results = 0

    def search_and_collect(questions):
        nonlocal overall_confidence, total_results
        for question in questions:
            results = vectorstore.similarity_search_with_score(question, k=3)#only top 3 will be considered by llama
            for doc, score in results:
                confidence = max(0, 1 - (score / 2.0))
                total_results += 1
                overall_confidence += confidence
                print(f"[Retrieval Agent] Chunk confidence: {round(confidence * 100, 1)}% | Source: {doc.metadata.get('source', 'unknown')}")
                if doc.page_content not in seen_contents:#this is bascially all the characters in chunks
                    all_chunks.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": round(score, 3),
                        "confidence": round(confidence * 100, 1)
                    })
                    seen_contents.add(doc.page_content)

    search_and_collect(sub_questions)
    avg_confidence = max([c["confidence"] / 100 for c in all_chunks]) if all_chunks else 0
    print(f"\n[Retrieval Agent] Average confidence: {round(avg_confidence * 100, 1)}%")

    if avg_confidence >= 0.50:# best case scenario
        print("[Retrieval Agent] High confidence — proceeding to synthesis")
        return {
            "chunks": all_chunks,
            "confidence": round(avg_confidence * 100, 1),
            "status": "success"
        }

    elif avg_confidence >= 0.30:#main crux of this code , here we tell llama bhai that if this condn then reanswer the qs to inc confidence
        print("[Retrieval Agent] Medium confidence — recalibrating...")
        original_confidence = avg_confidence
        rewrite_prompt = f"""Rephrase the following questions in a different way to improve search results.
Keep the meaning the same but use different words.

Questions:
{chr(10).join(sub_questions)}

Return the same number of rephrased questions, one per line, no extra text."""
        rewritten = llm.invoke(rewrite_prompt)
        rewritten_questions = [q.strip() for q in rewritten.strip().split("\n") if q.strip()]
        rewritten_questions = rewritten_questions[:2]
        print(f"[Retrieval Agent] Rewritten questions: {rewritten_questions}")
        search_and_collect(rewritten_questions)
        new_confidence = max([c["confidence"] / 100 for c in all_chunks]) if all_chunks else 0
        best_confidence= max(original_confidence,new_confidence)
        print(f"[Retrieval Agent] Original: {round(original_confidence*100,1)}% | After recalibration: {round(new_confidence*100,1)}% | Using: {round(best_confidence*100,1)}%")
        return {
            "chunks": all_chunks,
            "confidence": round(best_confidence * 100, 1),
            "status": "recalibrated"
        }

    else:
        print("[Retrieval Agent] Low confidence — insufficient data")
        return {
            "chunks": [],
            "confidence": round(avg_confidence * 100, 1),
            "status": "low_confidence"
        }

def synthesis_agent(query, retrieval_result, llm):
    print(f"\n[Synthesis Agent] Generating answer...")
    chunks = retrieval_result["chunks"]
    confidence = retrieval_result["confidence"]
    status = retrieval_result["status"]

    if status == "low_confidence" or not chunks:
        return {
            "answer": "Sorry, This question isnt relevant to the data provided to me. Please visit https://www.dell.com/support for further details.",
            "sources": [],
            "confidence": confidence,
            "status": "low_confidence"
        }

    context = ""
    sources = []
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['content']}\n"#making sure the indentation is fine
        source_name = chunk['source'].split("\\")[-1].split("/")[-1]#removing all the // from the path of files so that it looks clean
        if source_name not in sources:
            sources.append(source_name)

    recalibration_note = "Note: This answer was retrieved after query recalibration for better accuracy.\n" if status == "recalibrated" else ""

    prompt = f"""You are a helpful knowledge assistant for Dell Technologies.
Use ONLY the context provided below to answer the question.
If the context doesn't contain enough information, say so clearly.
Always be concise and accurate.
{recalibration_note}
Context:
{context}

Question: {query}

Answer:"""

    answer = llm.invoke(prompt)
    print(f"[Synthesis Agent] Answer generated with {confidence}% confidence")

    return {
        "answer": answer.strip(),
        "sources": sources,
        "confidence": confidence,
        "status": status
    }