from google.adk.agents.llm_agent import Agent
from chroma_loader import load_chroma



def get_resume_agent():
    agent = Agent(
        model="gemini-2.5-flash",
        instructions="""
You are a resume extraction agent.

Rules:
- Use ONLY the given context.
- Do NOT hallucinate.
- Return ONLY valid JSON.
- If a field is missing, return null.

Extract:
- full_name
- total_experience
- technical_skills
- education
- current_company
"""
    )
    return agent

vector_db = load_chroma()
agent = get_resume_agent()

query = "What is the candidate email address"

docs = vector_db.similarity_search(query, k=6)

context = "\n\n".join([doc.page_content for doc in docs])

result = agent.run(context)

print("✅ Final Output:\n", result)