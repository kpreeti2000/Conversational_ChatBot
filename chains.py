from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema.runnable import RunnableParallel
import re


HF_API_TOKEN = "TOKEN"

# Hugging Face LLM setup
mistral_llm = HuggingFaceHub(
    huggingfacehub_api_token=HF_API_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"max_new_tokens": 100}
)

llama_llm = HuggingFaceHub(
    huggingfacehub_api_token=HF_API_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"max_new_tokens": 100}
)

# Shared memory
memory = ConversationBufferMemory()

# Individual Chains
mistral_chain = ConversationChain(llm=mistral_llm, memory=memory)
llama_chain = ConversationChain(llm=llama_llm, memory=memory)

# Parallel execution (AutoChain alternative)
parallel_chains = RunnableParallel(mistral=mistral_chain, llama=llama_chain)


def combine_fn(responses):
    mistral_resp = responses["mistral"]["response"]
    llama_resp = responses["llama"]["response"]

    def extract_ai_reply(text):
        return text
        # match = re.search(r'(AI:.*?)(?:\n|$)', text)
        # return match.group(1).strip() if match else "Sorry, I couldn't find a clear response."

    mistral_reply = extract_ai_reply(mistral_resp)
    llama_reply = extract_ai_reply(llama_resp)

    # Pick the longest or identical responses
    return mistral_reply if mistral_reply == llama_reply else max([mistral_reply, llama_reply], key=len)

def chat():
    print("Multi-LLM Chatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        responses = parallel_chains.invoke(user_input)
        final_response = combine_fn(responses)

        print(f"Bot: {final_response}\n")

if __name__ == "__main__":
    chat()
