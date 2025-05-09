import os
import google.generativeai as genai
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables and configure APIs
load_dotenv()
genai.configure(api_key="Enter your API key here")
model = genai.GenerativeModel('gemini-1.5-flash')
weather = OpenWeatherMapAPIWrapper(openweathermap_api_key="Enter your API key here")
memory = ConversationBufferMemory(return_messages=True)

class ChatState(TypedDict):
    user_query: str
    is_weather_query: bool
    city: str | None
    weather_result: str | None
    final_response: str | None
    conversation_history: list

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini model."""
    return model.generate_content(prompt).text.strip()

def is_weather_query(query: str) -> bool:
    """Determine if the query is weather-related."""
    return get_gemini_response(f"Determine if this query is asking about weather. Answer with only 'yes' or 'no'. Query: {query}").lower() == 'yes'

def extract_city(query: str) -> str:
    """Extract city name from user query."""
    return get_gemini_response(f"Extract only the city name from this query. Return just the city name, nothing else. Query: {query}")

def get_weather_info(city: str) -> str:
    """Get weather information for the city."""
    try:
        return weather.run(city)
    except Exception as e:
        return f"Error getting weather for {city}: {str(e)}"

def format_weather_response(query: str, weather_info: str, conversation_history: list) -> str:
    """Format the weather response."""
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                            for msg in recent_history])
    
    prompt = f"""Context: {history_str}
    Query: {query}
    Weather Info: {weather_info}
    Provide a direct, concise weather response. If query is about a different city than the weather info, mention that."""
    
    return get_gemini_response(prompt)

def handle_general_conversation(query: str, conversation_history: list) -> str:
    """Handle general conversation queries."""
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                            for msg in recent_history])
    
    prompt = f"""You are a knowledgeable AI assistant with comprehensive information about current events, people, and positions. You have access to factual data including ages, dates, and biographical information.

    Context: {history_str}
    Query: {query}

    Rules:
    - Always provide complete, factual answers
    - For questions about people (ages, positions, etc.), include all known details
    - For follow-up questions, use context from previous messages
    - Never say "I don't have that information" for common factual data
    - Format responses as complete sentences with all relevant details
    - Keep responses concise but informative
    
    Now, provide a complete answer to the current query:"""
    
    return get_gemini_response(prompt)

# Node functions for the graph
def init_state(query: str) -> ChatState:
    return {
        "user_query": query,
        "is_weather_query": False,
        "city": None,
        "weather_result": None,
        "final_response": None,
        "conversation_history": memory.chat_memory.messages
    }

def check_query_type_node(state: ChatState) -> ChatState:
    state["is_weather_query"] = is_weather_query(state["user_query"])
    return state

def extract_city_node(state: ChatState) -> ChatState:
    if state["is_weather_query"]:
        state["city"] = extract_city(state["user_query"])
    return state

def get_weather_node(state: ChatState) -> ChatState:
    if state["is_weather_query"]:
        state["weather_result"] = get_weather_info(state["city"])
    return state

def format_response_node(state: ChatState) -> ChatState:
    if state["is_weather_query"]:
        state["final_response"] = format_weather_response(
            state["user_query"],
            state["weather_result"],
            state["conversation_history"]
        )
    else:
        state["final_response"] = handle_general_conversation(
            state["user_query"],
            state["conversation_history"]
        )
    
    memory.chat_memory.add_user_message(state["user_query"])
    memory.chat_memory.add_ai_message(state["final_response"])
    return state

# Build and compile the graph
graph = StateGraph(ChatState)
graph.add_node("check_query_type", RunnableLambda(check_query_type_node))
graph.add_node("extract_city", RunnableLambda(extract_city_node))
graph.add_node("get_weather", RunnableLambda(get_weather_node))
graph.add_node("format_response", RunnableLambda(format_response_node))

graph.add_edge("check_query_type", "extract_city")
graph.add_edge("extract_city", "get_weather")
graph.add_edge("get_weather", "format_response")
graph.add_edge("format_response", END)
graph.set_entry_point("check_query_type")

chat_executor = graph.compile()

def main():
    print("Welcome to the AI Assistant!")
    print("You can ask me about the weather or chat with me about anything.")
    print("I'll remember our conversation to provide better context-aware responses.")
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() == 'quit':
            break
            
        try:
            result = chat_executor.invoke(init_state(user_input))
            print("\n" + result["final_response"])
        except Exception as e:
            print(f"Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    main()
