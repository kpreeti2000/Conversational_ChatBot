import os
import requests
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated

# Set your OpenWeatherMap API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "e5bc857dfd67ee1ffc708b889da758b7")

# Define state schema
class WeatherState(TypedDict):
    city: str
    weather_result: str | None

# Tool to retrieve weather data
@tool
def get_weather(city: str) -> str:
    """Returns current weather for a given city."""
    url = (
        f"http://api.openweathermap.org/data/2.5/weather?q={city}"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return f"Error: {response.json().get('message', 'Failed to fetch weather.')}"
    
    data = response.json()
    temp = data['main']['temp']
    desc = data['weather'][0]['description']
    return f"The current temperature in {city} is {temp}°C with {desc}."

# Simple LangGraph state
def init_state(city: str) -> WeatherState:
    return {"city": city, "weather_result": None}

# Node function to call weather tool
def fetch_weather_node(state: WeatherState) -> WeatherState:
    city = state["city"]
    weather = get_weather.invoke(city)
    return {"city": city, "weather_result": weather}

# Build LangGraph
graph = StateGraph(WeatherState)
graph.add_node("get_weather", RunnableLambda(fetch_weather_node))
graph.set_entry_point("get_weather")
graph.add_edge("get_weather", END)
weather_executor = graph.compile()

# Execute
if __name__ == "__main__":
    city = input("Enter city name: ")
    result = weather_executor.invoke(init_state(city))
    print(result["weather_result"])

