from agent import planning_agent,retrieval_agent,synthesis_agent
from agent import load_vectorstore,get_llm

vectorstore=load_vectorstore()
llm=get_llm()
conversation_history = []
def run_pipeline(query):
    print("\n"+"="*50)
    print(f"Query: {query}")
    print("="*50)
    if conversation_history:
        history_text = "\n".join([
            f"User: {h['query']}\nAssistant: {h['answer']}"
            for h in conversation_history[-2:]  # only last 2 chats will be stored...
        ])
        enriched_query = f"""Previous conversation:
{history_text}

Current question: {query}

Answer the current question, using the conversation above only if it is relevant."""
    else:
        enriched_query = query
    sub_questions=planning_agent(enriched_query,llm)
    retrieval_result=retrieval_agent(sub_questions,vectorstore,llm)
    final_result=synthesis_agent(enriched_query,retrieval_result,llm)#since its history now query has to change to enriched query
    conversation_history.append({
        "query": query,
        "answer": final_result["answer"]
    })
    if len(conversation_history) > 2:#not letting the history extend more than 2.
        conversation_history.pop(0)
    print("\n"+"="*50)
    print("FINAL ANSWER:")
    print(final_result["answer"])
    print(f"\nsources: {final_result['sources']}")
    print(f"Confidence: {final_result['confidence']}%")
    print(f"Status: {final_result['status']}")
    print("="*50)

    return final_result
if __name__=="__main__":
    while True:
        query=input("\n Enter your question(or exit to quit):")
        if query.lower()=="exit":
            break
        run_pipeline(query)
