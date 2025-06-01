from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from tools import generate_image # Import the generate_image tool from tools.py

user_input = input("Input ('prompt' required): ")
if user_input:
    # Set up Ollama LLM
    llm = OllamaLLM(model="dolphin-mistral")
    
    # Register your tool
    tools = [generate_image]

    # Agent instructions to guide tool-calling
    agent_instructions = (
        "When you want to use a tool, respond with:\n"
        "Action:\n"
        "```{\"action\": \"<tool_name>\", \"action_input\":{...}}```"
        "Example:\n"
        "Input: 'Generate an image of a cat, steps 20, cfgScale 7.0'\n"
        "Thought: The user wants me to generate an image of a cat. I should use the generate_image tool\n"
        "Action:\n"
        "```{\"action\": \"generate_image\", \"action_input\": {\"prompt\": \"A cat\", \"steps\":20, \"cfgScale\":7.0}}```"
    )
    
    # Initialize agent with tool-calling, instructions, and 1 max_iteration
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iteration=1, #Prevents the agent from repeating
        agent_kwargs={"system_message": agent_instructions}
    )
    
    # Get the user input and invoke the agent
    result = agent.invoke({"input": user_input})
    print(result)