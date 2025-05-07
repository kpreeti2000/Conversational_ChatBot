import os
import requests
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import requests
from typing import Dict, List, Optional
# Configuration for different models
class ModelConfig:
    def __init__(self, name, api_url=None, api_token=None):
        self.name = name
        self.api_url = api_url
        self.api_token = api_token

# Model configurations
MODELS = {
    "mistralv2": ModelConfig(
        name="v.2",
        api_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        api_token="enter token here"
    ),
    "mistralv3": ModelConfig(
        name="v.3",
        api_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        api_token="enter token here"
    ),
}

class MultiModelChatbot:
    def __init__(self, default_model="mistralv2"):
        self.active_model = default_model
        self.shared_memory = []

    def _format_prompt(self, user_input):
        system_message = (
            "You are a helpful Chat-bot assistant. Keep answers very short and accurate."
            #"Be conversational. Talk."
            "When you don't know something, admit it rather than guessing."
        )
        prompt = f"<s>[INST] {system_message}\n\n"

        for msg in self.shared_memory:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"

        prompt += f"User: {user_input} [/INST]"
        return prompt

    def _query_model(self, user_input):
        model_config = MODELS[self.active_model]
        prompt = self._format_prompt(user_input)

        try:
            response = requests.post(
                model_config.api_url,
                headers={"Authorization": f"Bearer {model_config.api_token}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 100}}
            )

            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                text = result[0]['generated_text']
                # Corrected robust extraction logic:
                response_text = text.split("[/INST]")[-1].strip().split('\n')[0].strip()
                return response_text
            else:
                return "I encountered an error processing your request."
        except Exception as e:
            return f"Error: {str(e)}"

    def switch_model(self, model_name):
        if model_name in MODELS:
            self.active_model = model_name
            return f"Switched to {MODELS[model_name].name} model."
        else:
            return f"Model '{model_name}' not found. Using {MODELS[self.active_model].name}."

    def list_models(self):
        return "Available models: " + ", ".join(MODELS.keys())

    def process_message(self, user_input):
        if user_input.lower() == "list models":
            return self.list_models()

        if user_input.lower().startswith("use model "):
            model_name = user_input.lower().replace("use model ", "").strip()
            return self.switch_model(model_name)

        self.shared_memory.append({"role": "user", "content": user_input})
        response = self._query_model(user_input)
        self.shared_memory.append({"role": "assistant", "content": response})
        return response

    def clear_memory(self):
        self.shared_memory = []
        return "Conversation history cleared."

def chat():
    print("Multi-Model Chatbot ready! Type 'quit' to exit.")
    print("Special commands:")
    print("  'list models' - Show available models")
    print("  'use model [name]' - Switch to a different model")
    print("  'clear memory' - Clear conversation history for current model\n")

    chatbot = MultiModelChatbot()

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.lower() == "clear memory":
            print(f"Bot: {chatbot.clear_memory()}\n")
            continue

        response = chatbot.process_message(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    chat()
