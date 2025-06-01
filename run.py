from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from tools import generate_image # Import the generate_image tool from tools.py

# Set up Ollama LLM
llm = OllamaLLM(model="dolphin-mistral")

# Register your tool
tools = [generate_image]

# Agent instructions to guide tool-calling
agent_instructions = (
    "INSTRUCTIONS: You are an expert image generation assistant.\n"
    "Your available tool action is generate_image\n"
    "When you use the generate_image tool with a user input description of a scene, \n"
    "create a prompt with comma-separated words and short phrases describing the scene."
)

# Initialize agent with tool-calling, instructions, and 1 max_iteration
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iteration=1,
    agent_kwargs={"system_message":agent_instructions})

# Get the user prompt and invoke the agent
# The agent interprets all user input as a prompt for image generation
prompt = input("Prompt: ")
if prompt:
    result = agent.invoke({"input": prompt})
    print(result)