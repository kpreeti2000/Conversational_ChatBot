# LLM ChatBot- using langchain
Implemented a multiple model chatbot using LLM, LangChain and Hugging Face. It supports switching between multiple models(e.g.,GPT-4, Mistral 7B-Instruct) and maintain conversation history using BufferMemory. The chatbot uses PromptTemplate to define it's behavior and HuggingFaceEndpoint for querying models via APIs. Users can interact with chatbot, switch models and clear memory dynamically.

To install requests library:
use this command: pip install requests # which will let your code to make web requests(like talking to an online API)
