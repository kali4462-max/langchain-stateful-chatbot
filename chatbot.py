import os
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY not found in .env file!")
    exit()

def get_working_model():
    client = Groq(api_key=api_key)
    models = [m.id for m in client.models.list().data]
    
    # Filter only standard text chat models
    chat_models = [
        m for m in models 
        if any(k in m.lower() for k in ["llama", "qwen", "gemma", "mixtral", "deepseek"])
        and "guard" not in m.lower() and "vision" not in m.lower()
    ]
    
    for model_id in chat_models + models:
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2
            )
            return model_id
        except Exception:
            continue
    raise RuntimeError("No active chat model found on this Groq account.")

active_model = get_working_model()

llm = ChatGroq(
    api_key=api_key,
    model=active_model,
    temperature=0.3
)

# Task 2: Prompt template with MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Answer questions clearly using the conversation history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

# Task 3: List to store conversation history
chat_history = []

# Task 4: Function to trim older messages
def trim_history(history, max_len=6):
    if len(history) > max_len:
        return history[-max_len:]
    return history

# Task 5 & 6: Q&A function with history
def chat(user_input):
    global chat_history
    chat_history = trim_history(chat_history, max_len=6)
    
    response = chain.invoke({
        "history": chat_history,
        "question": user_input
    })
    
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    
    return response.content

if __name__ == "__main__":
    sample_questions = [
        "Explain Python lists in brief",
        "Give an example",
        "What about tuples?"
    ]
    
    print(f"--- Using Model: {active_model} ---")
    print("--- Sample Test ---")
    for q in sample_questions:
        print(f"\nUser: {q}")
        ans = chat(q)
        print(f"Bot: {ans}")
        
    print("\n--- Start Chatting (type 'exit' to stop) ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue
            
        reply = chat(query)
        print(f"Bot: {reply}")