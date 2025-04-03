import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import requests
from typing import Dict, List, Optional

# Configuration for different models
class ModelConfig:
    def __init__(self, name, api_url=None, api_token=None, is_local=False):
        self.name = name
        self.api_url = api_url
        self.api_token = api_token
        self.is_local = is_local

# Model configurations
MODELS = {
    "mistral": ModelConfig(
        name="Mistral-7B-Instruct",
        api_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        api_token="enter token here"  # Replace with your actual token
    ),
    "llama": ModelConfig(
        name="Llama-2-7b-chat",
        api_url="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
        api_token="enter token here"  # Replace with your actual token
    ),
    # Add more models as needed
}

class MultiModelChatbot:
    def __init__(self, default_model="mistral"):
        self.default_model = default_model
        self.active_model = default_model
        self.memories = {}
        
        # Initialize memory for each model
        for model_key in MODELS.keys():
            self.memories[model_key] = []

    def _format_messages_for_api(self, model_key, user_input):
        """Format messages for API call with chat history"""
        system_message = "You are a helpful assistant that provides accurate and concise answers. When you don't know something, admit it rather than guessing."
        
        # Prepare formatted prompt with history
        formatted_prompt = f"<s>[INST] {system_message}\n\n"
        
        # Add conversation history
        for msg in self.memories[model_key]:
            if msg["role"] == "user":
                formatted_prompt += f"User: {msg['content']}\n"
            else:
                formatted_prompt += f"Assistant: {msg['content']}\n"
        
        # Add current user input
        formatted_prompt += f"User: {user_input} [/INST]"
        
        return formatted_prompt
    
    def _query_model(self, model_key, user_input):
        """Query the model API with formatted prompt"""
        model_config = MODELS[model_key]
        formatted_prompt = self._format_messages_for_api(model_key, user_input)
        
        try:
            # Query the API
            response = requests.post(
                model_config.api_url,
                headers={"Authorization": f"Bearer {model_config.api_token}"},
                json={"inputs": formatted_prompt}
            )
            
            # Process response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0]['generated_text']
                # Extract just the assistant's response
                if "Assistant:" in text.split(formatted_prompt)[-1]:
                    response_text = text.split(formatted_prompt)[-1].split("Assistant:")[-1].strip()
                else:
                    response_text = text.split(formatted_prompt)[-1].strip()
                return response_text
            else:
                return "I encountered an error processing your request."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def switch_model(self, model_name):
        """Switch to a different model"""
        if model_name in MODELS:
            self.active_model = model_name
            return f"Switched to {MODELS[model_name].name} model."
        else:
            return f"Model '{model_name}' not found. Using {MODELS[self.active_model].name}."
    
    def list_models(self):
        """List all available models"""
        return "Available models: " + ", ".join(MODELS.keys())
    
    def process_message(self, user_input):
        """Process user message and return response"""
        # Check for special commands
        if user_input.lower() == "list models":
            return self.list_models()
        
        if user_input.lower().startswith("use model "):
            model_name = user_input.lower().replace("use model ", "").strip()
            return self.switch_model(model_name)
        
        # Process with active model
        try:
            # Add user message to memory
            self.memories[self.active_model].append({"role": "user", "content": user_input})
            
            # Get response from the current model
            response = self._query_model(self.active_model, user_input)
            
            # Add assistant response to memory
            self.memories[self.active_model].append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_memory(self, model_key=None):
        """Clear memory for specified model or active model if not specified"""
        if model_key is None:
            model_key = self.active_model
            
        if model_key in self.memories:
            self.memories[model_key] = []
            return f"Memory cleared for {MODELS[model_key].name} model."
        else:
            return f"Model '{model_key}' not found."

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
            
        # Process the message
        response = chatbot.process_message(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    chat()