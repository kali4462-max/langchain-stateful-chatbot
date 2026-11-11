# LangChain Stateful Chatbot with Conversation History

A stateful Q&A chatbot built using **LangChain** and **Groq API** that remembers past interactions, uses `MessagesPlaceholder` for dynamic context injection, and automatically trims older messages to prevent token overflow.

## Tech Stack
* **Language:** Python
* **Framework:** LangChain (`langchain-core`, `langchain-groq`)
* **LLM Provider:** Groq API
* **Environment Management:** `python-dotenv`

## Project Structure
```text
LangChain_Stateful_Chatbot/
├── chatbot.py         # Main stateful chatbot implementation
├── requirements.txt   # Required Python dependencies
├── .gitignore         # Ignored files (.env, __pycache__)
└── README.md          # Project documentation & conceptual answers