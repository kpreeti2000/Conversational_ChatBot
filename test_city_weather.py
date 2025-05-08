import os
import google.generativeai as genai
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = "AIzaSyDyrMo3bhaPMvtv2qrY0TfflXlz3-bGkKg"
OPENWEATHERMAP_API_KEY = "e5bc857dfd67ee1ffc708b889da758b7"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure OpenWeatherMap
weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=OPENWEATHERMAP_API_KEY)

# Define state schema
class WeatherState(TypedDict):
    user_query: str
    city: str | None
    weather_result: str | None
    final_response: str | None

def extract_city(query: str) -> str:
    """Extract city name from user query using Gemini."""
    prompt = f"""Extract only the city name from this query. Return just the city name, nothing else.
    Query: {query}"""
    
    response = model.generate_content(prompt)
    return response.text.strip()

def get_weather_info(city: str) -> str:
    """Get weather information for the city."""
    try:
        return weather.run(city)
    except Exception as e:
        return f"Error getting weather for {city}: {str(e)}"

def format_response(query: str, weather_info: str) -> str:
    """Format the final response using Gemini."""
    prompt = f"""Based on the user's query and the weather information, provide a natural, conversational response.
    User Query: {query}
    Weather Information: {weather_info}
    Provide a friendly, concise response that directly answers the user's question."""
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Node functions for the graph
def init_state(query: str) -> WeatherState:
    return {
        "user_query": query,
        "city": None,
        "weather_result": None,
        "final_response": None
    }

def extract_city_node(state: WeatherState) -> WeatherState:
    state["city"] = extract_city(state["user_query"])
    return state

def get_weather_node(state: WeatherState) -> WeatherState:
    state["weather_result"] = get_weather_info(state["city"])
    return state

def format_response_node(state: WeatherState) -> WeatherState:
    state["final_response"] = format_response(
        state["user_query"],
        state["weather_result"]
    )
    return state

# Build the graph
graph = StateGraph(WeatherState)

# Add nodes
graph.add_node("extract_city", RunnableLambda(extract_city_node))
graph.add_node("get_weather", RunnableLambda(get_weather_node))
graph.add_node("format_response", RunnableLambda(format_response_node))

# Add edges
graph.add_edge("extract_city", "get_weather")
graph.add_edge("get_weather", "format_response")
graph.add_edge("format_response", END)

# Set entry point
graph.set_entry_point("extract_city")

# Compile the graph
weather_executor = graph.compile()

def main():
    print("Welcome to the Weather Assistant!")
    print("Ask me about the weather in any city (e.g., 'What's the temperature in London?')")
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() == 'quit':
            break
            
        try:
            result = weather_executor.invoke(init_state(user_input))
            print("\n" + result["final_response"])
        except Exception as e:
            print(f"Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    main()