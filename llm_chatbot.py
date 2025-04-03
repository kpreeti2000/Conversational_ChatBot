import os
from langchain.llms import HuggingFaceHub
from langchain import LLMChain, PromptTemplate
from autochain import AutoChain  # Ensure autochain is installed

# Replace with your actual HuggingFace API token
HF_API_TOKEN = "enter token here"

# Define prompt template used by both models
prompt_template = (
    "You are a helpful assistant that provides accurate and concise answers. "
    "User: {user_input} Assistant:"
)
template = PromptTemplate(input_variables=["user_input"], template=prompt_template)

# Instantiate LLMs using HuggingFaceHub for Mistral and Llama models
mistral_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=HF_API_TOKEN,
    model_kwargs={"prompt_format": "mistral"}
)
llama_llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    huggingfacehub_api_token=HF_API_TOKEN,
    model_kwargs={"prompt_format": "llama"}
)

# Create LLMChains for each model
mistral_chain = LLMChain(llm=mistral_llm, prompt=template)
llama_chain = LLMChain(llm=llama_llm, prompt=template)

def combine_responses(responses):
    """Combine responses from both models using a simple heuristic."""
    responses = [resp.strip() for resp in responses]
    # If both responses are identical, return one.
    if responses[0] == responses[1]:
        return responses[0]
    # Prefer a response containing the current year (e.g., "2025")
    for resp in responses:
        if "2025" in resp:
            return resp
    # Otherwise, default to the first response.
    return responses[0]

class MultiLLMChatbot:
    def __init__(self):
        # AutoChain queries both chains and then applies the combine function.
        self.auto_chain = AutoChain(chains=[mistral_chain, llama_chain], combine_fn=combine_responses)

    def process_message(self, user_input: str) -> str:
        # Run the AutoChain with the current user input.
        return self.auto_chain.run({"user_input": user_input})

def chat():
    print("Chatbot ready! Type 'quit' to exit.\n")
    bot = MultiLLMChatbot()
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        response = bot.process_message(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    chat()
