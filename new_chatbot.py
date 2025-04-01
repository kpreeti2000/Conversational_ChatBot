from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# --- Step 1: Set up multiple LLM models ---
# Create two LLM objects using Hugging Face Hub.
# Replace "YOUR_API_TOKEN" with your Hugging Face API token.
llm_mistral = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token="Enter Token Here"
)
llm_flan = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token="Enter Token Here"
)

# Store models in a dictionary so you can choose later.
llms = {
    "mistral": llm_mistral,
    "flan": llm_flan
}

# Select the desired model. (You can add logic to switch dynamically.)
selected_model = "mistral"
llm = llms[selected_model]

# --- Step 2: Integrate Memory ---
# ConversationBufferMemory stores all previous conversation turns.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Step 3: Create a Conversation Chain ---
# The ConversationChain automatically chains the LLM calls together with memory.
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

def chat():
    print("Chatbot ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        # Use the conversation chain which includes memory.
        response = conversation.predict(input=user_input)
        print("Bot:", response, "\n")

if __name__ == "__main__":
    chat()
