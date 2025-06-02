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
    enhanced_prompt: str
    image_type_category: str

class SwarmUIAgent:
    def __init__(self, model_name: str = "dolphin-mistral"):
        self.llm = OllamaLLM(model=model_name)
        self.tools = [generate_image]
        self.graph = self._build_graph()

        # Define parameter presets for different image types
        self.parameter_presets = {
            "anime": {
                "width": 1024,
                "height": 1024,
                "cfgScale": 4,
                "steps": 20,
                "sampler": "euler",
                "scheduler": "simple"
            },
            "realistic": {
                "width": 1024,
                "height": 1024,
                "cfgScale": 3,
                "steps": 50,
                "sampler": "dpmpp_3m_sde_gpu",
                "scheduler": "simple"
            }
        }

    def _build_graph(self):
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_input", self.analyze_input)
        workflow.add_node("natural_response", self.natural_response)
        workflow.add_node("enhance_prompt", self.enhance_prompt)
        workflow.add_node("determine_parameters", self.determine_parameters)
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
                "generate_image":"enhance_prompt"
            }
        )

        # Add paths to end
        #Natural response path
        workflow.add_edge("natural_response", END)

        #Image generation path
        workflow.add_edge("enhance_prompt", "determine_parameters")
        workflow.add_edge("determine_parameters", "prepare_tool_call")
        workflow.add_edge("prepare_tool_call", "execute_tool")
        workflow.add_edge("execute_tool", "format_tool_response")
        workflow.add_edge("format_tool_response", END)

        return workflow.compile()

    def analyze_input(self, state: AgentState) -> AgentState:
        """Analyze user input to determine if image generation is needed"""
        user_input = state["user_input"].lower()

        # Keywords that suggest image generation
        image_keywords = [
            "generate", "realistic", "anime", "photo", "cartoon"
        ]

        needs_image = any(keyword in user_input for keyword in image_keywords)

        # Extract explicit parameters from user input (these will override presets)
        explicit_params = self.extract_explicit_params(state["user_input"])

        return {
            **state,
            "needs_image_generation": needs_image,
            "image_params": explicit_params,
            "enhanced_prompt": "",
            "image_type_category": ""
        }

    def extract_explicit_params(self, user_input: str) -> dict:
        """Extract explicitly specified image generation parameters from user input"""
        params = {}
        
        # Extract numerical parameters
        param_patterns = {
            'steps': r'steps?\s*:?\s*(\d+)',
            'cfgScale': r'cfg(?:\s*scale)?\s*:?\s*([\d.]+)',
            'seed': r'seed\s*:?\s*(-?\d+)',
            'images': r'images?\s*:?\s*(\d+)',
            'width': r'width\s*:?\s*(\d+)',
            'height': r'height\s*:?\s*(\d+)',
            'refinerControlPercentage': r'refinerControlPercentage?\s*:?\s*(\d+)',
            'refinerUpscale': r'refinerUpscale?\s*:?\s*(\d+)',
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

    def enhance_prompt(self, state: AgentState) -> AgentState:
        """Enhance the user's prompt for better image generation"""
        user_input = state["user_input"]
        
        enhancement_prompt = f"""
User_Request: '{user_input}'

Take the User_Request and create a comma-separated Stable Diffusion image generation prompt from it using short descriptive words and phrases.
tag lists. The prompt should accurately and clearly describe the user's request while enhancing it with artistic details and technical
quality terms relevent to the indicated style (anime or realism). Follow the guidelines below while creating the prompt:

Guidelines for prompt generation:
1. Use 1-3 word phrases separated by commas. Do not wrap phrases in quotes or any other characters.
2. Do not wrap the prompt in quotes or any other characters.
3. Do not repeat words anywhere within the prompt and keep the number of words less than 120.
4. Keep the core intent of the user's request unchanged. Add detail and context to missing prompt elements.
6. Organize tags within the prompt, such as scene layout tags, character description tags, and background description tags.
7. If the user requests to "avoid" or "not include" certain features, DO NOT INCLUDE those features in the prompt.

Prompt Format Example: <tag>, <tag>, <tag phrase>, <tag phrase>, <tag>, <tag>

Enhanced prompt:"""

        enhanced_prompt = self.llm.invoke(enhancement_prompt).strip()
        
        return {
            **state,
            "enhanced_prompt": enhanced_prompt
        }

    def determine_parameters(self, state: AgentState) -> AgentState:
        """Determine optimal image generation parameters based on the prompt and request type"""
        user_input = state["user_input"].lower()
        enhanced_prompt = state["enhanced_prompt"].lower()
        
        # Categorize the image type based on keywords
        category_keywords = {
            "realistic": ["realism", "realistic", "photoreal", "photorealism", "real", "real life"],
            "anime": ["anime", "cartoon", "manga"]
        }
        
        detected_category = "anime"
        for category, keywords in category_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                detected_category = category
                break
        
        # Get base parameters for the detected category
        base_params = self.parameter_presets[detected_category].copy()
        
        # Override with any explicitly specified parameters
        final_params = {**base_params, **state["image_params"]}
        
        # Set the enhanced prompt as the main prompt
        final_params["prompt"] = state["enhanced_prompt"]
        
        return {
            **state,
            "image_params": final_params,
            "image_type_category": detected_category
        }

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
        tool_call = {
            "name": "generate_image",
            "args": state["image_params"],
            "id": "generate_image_call_1"
        }
        
        ai_message = AIMessage(
            content="I'll generate that image for you using parameters for {state['image_type_category']} style.",
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
                    response += f"🎨 Enhanced prompt: \"{state['enhanced_prompt']}\"\n"
                    response += f"🎯 Detected style: {state['image_type_category']}\n"
                    response += f"📸 Generated {len(result['images'])} image(s)\n"
                    
                    # Show key parameters used
                    params = state['image_params']
                    response += f"⚙️ Parameters: {params.get('width', 'N/A')}x{params.get('height', 'N/A')}, "
                    response += f"Steps: {params.get('steps', 'N/A')}, CFG: {params.get('cfgScale', 'N/A')}\n"
                    
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
            "image_params": {},
            "enhanced_prompt": "",
            "image_type_category": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result

    def get_available_categories(self) -> list:
        """Return list of available image categories and their descriptions"""
        categories = {
            "realistic": "High steps, low cfg, optimized for generating realistic images",
            "anime": "(Default) Low steps, high cfg, optimized for quickly generating anime images",
        }
        return categories

if __name__ == "__main__":
    agent = SwarmUIAgent()
    
    print("SwarmUI Agent with Intelligent Prompt Enhancement")
    print("Available image categories:", ", ".join(agent.get_available_categories().keys()))
    print("Examples:")
    print("- 'Generate a fantasy landscape with dragons'") 
    print("- 'Generate an anime character with blue hair'")
    print("- 'Generate a photorealistic cat sitting in sunlight'")

    while True:
        user_input = input("\nInput ('quit' to exit): ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if user_input.strip():
            try:
                result = agent.run(user_input)
                print(f"\n{result['messages'][-1].content}")
                    
            except Exception as e:
                print(f"Error: {e}")