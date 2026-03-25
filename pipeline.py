from agent import planning_agent,retrieval_agent,synthesis_agent
from agent import load_vectorstore,get_llm

def run_pipeline(query):
    print("\n"+"="*50)
    print(f"Query: {query}")
    print("="*50)
    vectorstore=load_vectorstore()
    llm=get_llm()

    sub_questions=planning_agent(query,llm)
    retrieval_result=retrieval_agent(sub_questions,vectorstore,llm)
    final_result=synthesis_agent(query,retrieval_result,llm)
    print("\n"+"="*50)
    print("FINAL ANSWER:")
    print(final_result["answer"])
    print(f"\nsources: {final_result['sources']}")
    print(f"Confidence: {final_result['confidence']}%")
    print(f"Status: {final_result['status']}")
    print("="*50)

    return final_result
if __name__=="__main__":
    query=input("Enter your question:")
    run_pipeline(query)