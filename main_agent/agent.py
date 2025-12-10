from google.adk.agents.llm_agent import Agent
from datetime import datetime
from google.adk.tools import FunctionTool ,google_search_tool
fr
#tool1: get current time

def current_time()->dict:
     """
     Get the city name and give the time.
     Returns:
         dict: A dictionary containing the current date and time in the format "YYYY-MM-DD HH:MM:SS".
     
     """
     from datetime import datetime
     return {"data time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

time_tool = FunctionTool(func=current_time)

# root_agent = Agent(
#     model="gemini-2.5-flash",
#     name='root_agent',
#     description='A helpful assistant for user questions.',
#     instruction='You are a helpful agent ask the user their name and greet them warmly.',
# )


root_agent = Agent(
    model="gemini-2.5-flash",
    name='root_agent',
    description='You are data time telling agent.',
    instruction='You will ask for for which location the user wants to know the current date and time, and then provide the current date and time for that location using the current_time tool.',
    tools=[current_time],
)
