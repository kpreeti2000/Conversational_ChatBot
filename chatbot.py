import requests

# Replace with your Hugging Face API token
API_TOKEN = "Hugging face token"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()

def chat():
    print("Chatbot ready! Type 'quit' to exit.\n")
    # Define a set of greetings
    greetings = {"hi", "hello", "hey", "howdy"}
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        # If the user input is a greeting, reply directly without calling the API.
        if user_input.lower() in greetings:
            print("Bot: Hello!\n")
            continue

        # Build prompt with clear instructions for a single Q&A turn
        prompt = (
            "You are a helpful assistant. Answer the following question concisely "
            "without asking any follow-up questions or adding extra commentary. Only provide a direct answer.\n\n"
            f"User: {user_input}\nBot: "
        )
        
        output = query(prompt)
        try:
            bot_response = output[0]['generated_text'].strip()
            # Remove repeated prompt instructions if present.
            if bot_response.startswith("You are a helpful assistant."):
                bot_response = bot_response.split("\n")[-1].strip()
            if "Bot:" in bot_response:
                bot_response = bot_response.split("Bot:")[-1].strip()
        except Exception as e:
            bot_response = "Sorry, I couldn't get a response."
        
        print("Bot:", bot_response, "\n")

if __name__ == "__main__":
    chat()
