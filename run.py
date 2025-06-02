from typing import TypedDict, Annotated, Literal
import operator
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import generate_image # Import the generate_image tool from tools.py
import re
import json

#Define agent state structure
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_input: str
    needs_image_generation: bool
    image_params: dict

class SwarmUIAgent:
    def __init__(self, model_name: str = "dolphin-mistral"):
        self.llm = OllamaLLM(model=model_name)
        self.tools = [generate_image]
        self.graph = self._build_graph()

    def _build_graph(self):
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_input", self.analyze_input)
        workflow.add_node("natural_response", self.natural_response)
        workflow.add_node("prepare_tool_call", self.prepare_tool_call)
        workflow.add_node("execute_tool", ToolNode(self.tools))
        workflow.add_node("format_tool_response", self.format_tool_response)

        # Define entry
        workflow.set_entry_point("analyze_input")

        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_input",
            self.route_response,
            {
                "natural":"natural_response",
                "generate_image":"prepare_tool_call"
            }
        )

        # Add paths to end
        #Natural response path
        workflow.add_edge("natural_response", END)

        #Image generation path
        workflow.add_edge("prepare_tool_call", "execute_tool")
        workflow.add_edge("execute_tool", "format_tool_response")
        workflow.add_edge("format_tool_response", END)

        return workflow.compile()

    def analyze_input(self, state: AgentState) -> AgentState:
        """Analyze user input to determine if image generation is needed"""
        user_input = state["user_input"].lower()

        # Keywords that suggest image generation
        image_keywords = [
            "generate", "create", "make", "draw", "image", "picture", 
            "photo", "illustration", "art", "painting", "render"
        ]

        needs_image = any(keyword in user_input for keyword in image_keywords)

        # Extract parameters from user input
        image_params = self.extract_image_params(state["user_input"])

        return {
            **state,
            "needs_image_generation": needs_image,
            "image_params": image_params
        }

    def extract_image_params(self, user_input: str) -> dict:
        """Extract image generation parameters from user input"""
        params = {}
        
        # Extract numerical parameters
        param_patterns = {
            'steps': r'steps?\s*:?\s*(\d+)',
            'cfgScale': r'cfg(?:\s*scale)?\s*:?\s*([\d.]+)',
            'seed': r'seed\s*:?\s*(-?\d+)',
            'images': r'images?\s*:?\s*(\d+)',
            'width': r'width\s*:?\s*(\d+)',
            'height': r'height\s*:?\s*(\d+)',
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                value = match.group(1)
                if param in ['cfgScale', 'refinerControlPercentage', 'refinerUpscale']:
                    params[param] = float(value)
                else:
                    params[param] = int(value)
        
        # Extract string parameters
        string_patterns = {
            'prompt': r'prompt\s*:?\s*["\']([^"\']+)["\']',
            'negative': r'negative\s*:?\s*["\']([^"\']+)["\']',
            'model': r'model\s*:?\s*["\']?([^"\']+)["\']?',
            'sampler': r'sampler\s*:?\s*["\']?([^"\']+)["\']?',
            'scheduler': r'scheduler\s*:?\s*["\']?([^"\']+)["\']?',
        }
        
        for param, pattern in string_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                params[param] = match.group(1).strip()

        return params

    def route_response(self, state: AgentState) -> Literal["natural", "generate_image"]:
        """Route to appropriate response type based on analysis"""
        return "generate_image" if state["needs_image_generation"] else "natural"

    def natural_response(self, state: AgentState) -> AgentState:
        """Generate a natural language response"""
        # Create a simple prompt for natural conversation
        prompt = f"Respond naturally to this user input: {state['user_input']}"
        
        response = self.llm.invoke(prompt)
        
        return {
            **state,
            "messages": [AIMessage(content=response)]
        }

    def prepare_tool_call(self, state: AgentState) -> AgentState:
        """Prepare the tool call message for execution"""
        # Use the extracted parameters, with defaults for missing ones
        tool_input = {
            **state["image_params"]  # Include any extracted parameters
        }
        
        tool_call = {
            "name": "generate_image",
            "args": tool_input,
            "id": "generate_image_call_1"
        }
        
        ai_message = AIMessage(
            content="I'll generate that image for you.",
            tool_calls=[tool_call]
        )
        
        return {
            **state,
            "messages": [ai_message]
        }

    def format_tool_response(self, state: AgentState) -> AgentState:
        """Format the tool response for display"""
        # Get the last message (should be a ToolMessage from the tool execution)
        last_message = state["messages"][-1]
        
        if isinstance(last_message, ToolMessage):
            # The tool has been executed, now format the response
            try:
                # Parse the tool result
                result = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content
                
                # Create a user-friendly response
                if 'images' in result and result['images']:
                    response = f"✅ Image generated successfully!\n"
                    response += f"📸 Generated {len(result['images'])} image(s)\n"
                    for i, img_url in enumerate(result['images'], 1):
                        response += f"🔗 Image {i}: {img_url}\n"
                else:
                    response = "❌ Image generation failed - no images returned"
                    
            except Exception as e:
                response = f"❌ Error processing image generation result: {str(e)}"
        else:
            response = "❌ Unexpected response format from image generation tool"
        
        return {
            **state,
            "messages": [AIMessage(content=response)]
        }

    def run(self, user_input: str) -> dict:
        """Run the agent with user input"""
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "needs_image_generation": False,
            "image_params": {}
        }
        
        result = self.graph.invoke(initial_state)
        return result

if __name__ == "__main__":
    agent = SwarmUIAgent()
    
    while True:
        user_input = input("\nInput ('quit' to exit): ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if user_input.strip():
            try:
                result = agent.run(user_input)
                print(f"{result['messages'][-1].content}")
                    
            except Exception as e:
                print(f"Error: {e}")