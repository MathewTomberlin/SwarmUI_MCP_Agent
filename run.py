from typing import TypedDict, Annotated, Literal
import operator
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import generate_image # Import the generate_image tool from tools.py
import re
import json
import yaml
import os
import streamlit as st
import requests
import base64
import time

def make_spinner(text = "In progress..."):
    with st.spinner(text):
        yield
sp = iter(make_spinner())

def extract_base64_from_data_uri(data_uri: str) -> str:
    # Extract base64 part from data URI
    match = re.match(r"data:.*?;base64,(.*)", data_uri)
    if not match:
        raise ValueError("Invalid data URI format")
    return match.group(1)

def get_image_data_uri(img_url: str, api_base_url: str) -> str:
    if img_url.startswith("data:"):
        return img_url  # Already a data URI
    # Otherwise, fetch the image from the API
    # Ensure the URL is absolute
    if not img_url.startswith("http"):
        img_url = api_base_url.rstrip("/") + "/" + img_url.lstrip("/")
    resp = requests.get(img_url)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type", "image/png")  # Default to PNG
    b64 = base64.b64encode(resp.content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

#Define agent state structure
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_input: str
    needs_image_generation: bool
    image_params: dict
    enhanced_prompt: str
    image_type_category: str
    explicit_prompt_type: str #'none', 'explicit', 'enhance'

class SwarmUIAgent:
    def __init__(self, model_name: str = "dolphin-mistral"):
        st.title("SwarmUI Agent")
        with st.expander("Settings", expanded=False):
            self.port = st.text_input("SwarmUI API Port", value="7801")
            self.llm_model = st.selectbox("LLM Model",["dolphin-mistral", "dolphin-llama3"])
            self.vision_model = st.selectbox("Vision Model", ["gemma3:12b", "llava","bakllava","qwen-vl", "None"])
        self.llm = OllamaLLM(model=self.llm_model)
        self.vision_llm = OllamaLLM(model=self.vision_model)
        self.tools = [generate_image]
        self.graph = self._build_graph()
        self.categories = self.load_image_generation_categories()
        self.output = st.empty()
        self.loading = st.empty()
        self.input = st.text_area("User Input",
                                   help="Enter your request to the agent. Mention 'generate' to trigger SwarmUI image generation.",
                                   placeholder="Enter your request for the agent. Mention 'generate' for SwarmUI image generation.")
        self.submit = st.button("Submit")
        if "welcome_shown" not in st.session_state:
            with self.output.container():
                self.output.write(
                    f"""
                    Hello! I'm an agent that can help you generate images using your local SwarmUI API\n
                    **Setting Categories**: NONE, {", ".join(list(self.get_available_categories().keys())[1:])}\n
                    **Example**: Generate an anime image of a character with blue hair
                    """
                )

        if self.submit:
            try:
                st.session_state["welcome_shown"] = True
                result = self.run(self.input)
                # Accumulate all AI messages as a single string, separated by double newlines for clarity
                all_msgs = []
                for msg in result['messages']:
                    if isinstance(msg, AIMessage):
                        all_msgs.append(msg.content)
                
                # Join all messages and display with st.markdown for multiline support
                self.output.markdown('\n\n'.join(all_msgs).replace('\n', '  \n'))
            except Exception as e:
                self.output.write(f"Error: {e}")

    def load_image_generation_categories(self, config_path: str = "image_generation_categories.yaml") -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["categories"]

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
        # this starts the spinning
        next(sp)

        # Keywords that suggest image generation
        image_keywords = [
            "generate", "realistic", "anime", "photo", "cartoon"
        ]

        needs_image = any(keyword in user_input for keyword in image_keywords)

        # Extract explicit parameters from user input (these will override presets)
        explicit_params, explicit_prompt_type = self.extract_explicit_params(state["user_input"])

        return {
            **state,
            "needs_image_generation": needs_image,
            "image_params": explicit_params,
            "enhanced_prompt": "",
            "image_type_category": "",
            "explicit_prompt_type": explicit_prompt_type
        }

    def extract_explicit_params(self, user_input: str) -> dict:
        """Extract explicitly specified image generation parameters from user input"""
        params = {}
        explicit_prompt_type = "none"
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

        prompt = None
        match = re.search(r'prompt\s*:?\s*["\']([^"\']+)["\']', user_input, re.IGNORECASE)
        if match:
            prompt = match.group(1).strip()
            params['prompt'] = prompt
            # Detect if prompt is already enhanced
            if prompt.lower().startswith("enhance:"):
                explicit_prompt_type = "enhance"
            else:
                explicit_prompt_type = "explicit"

        return params, explicit_prompt_type

    def route_response(self, state: AgentState) -> Literal["natural", "generate_image"]:
        """Route to appropriate response type based on analysis"""
        return "generate_image" if state["needs_image_generation"] else "natural"

    def enhance_prompt(self, state: AgentState) -> AgentState:
        """Enhance the user's prompt for better image generation"""
        if state.get("explicit_prompt_type") == "explicit":
            # Use the explicit prompt as the enhanced prompt
            return {
                **state,
                "enhanced_prompt": state["image_params"].get("prompt", "")
            }

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
        self.output.write(
            f"""\n
            Generating image with SwarmUI...\n\n
            📝 **Base Prompt**: {user_input}\n
            ✨ **Enhanced Prompt**: {enhanced_prompt}
            """
        )

        return {
            **state,
            "enhanced_prompt": enhanced_prompt
        }

    def determine_parameters(self, state: AgentState) -> AgentState:
        """Determine optimal image generation parameters based on the prompt and request type"""
        user_input = state["user_input"].lower()
        enhanced_prompt = state["enhanced_prompt"].lower()
        
        # Reload categories from YAML each time
        self.categories = self.load_image_generation_categories()

        detected_category = None
        for category, data in self.categories.items():
            if any(keyword in user_input for keyword in data.get("keywords", [])):
                detected_category = category
                break
        if not detected_category:
            detected_category = next(iter(self.categories))  # fallback to first category

        base_params = self.categories[detected_category]["parameters"].copy()
        
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
            content="",
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
        api_base_url = f"http://localhost:{self.port}"  # Change to your SwarmUI API base URL

        if isinstance(last_message, ToolMessage):
            # The tool has been executed, now format the response
            try:
                # Parse the tool result
                result = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content
                
                # Create a user-friendly response
                if 'images' in result and result['images']:
                    response = f"✅ **Image generated successfully!**\n\n"
                    response += f"🎨 {'**Enhanced Prompt**' if state['explicit_prompt_type'] != 'explicit' else '**Prompt**'}: {state['enhanced_prompt']}\n"
                    response += f"🎯 **Style**: {state['image_type_category']}\n"
                    response += f"📸 **Generated** {len(result['images'])} image(s)\n"
                    
                    # Show key parameters used
                    params = state['image_params']
                    response += f"⚙️ **Parameters**: *{params.get('model','INVALID MODEL')}*, "
                    response += f"**{params.get('width', '-')}x{params.get('height', '-')}**, "
                    response += f"**Steps**: {params.get('steps', 'INVALID STEPS')}, **CFG**: {params.get('cfgScale', 'INVALID CFG')}\n"
                    
                    for i, img_url in enumerate(result['images'], 1):
                        data_uri = get_image_data_uri(img_url, api_base_url)
                        st.image(data_uri, caption=f"Image {i}")
                        response += f"🔗 **Image {i}**: {img_url}\n"

                        if self.vision_model != "None":
                            # Extract base64 string for vision model
                            base64_str = extract_base64_from_data_uri(data_uri)
                            description_prompt = "Describe this image in detail. Be as accurate and precise as possible. If a detail is too small or blurry to clearly understand, ignore it."
                            vision_response = self.vision_llm.invoke(
                                input=description_prompt,
                                images=[base64_str]
                            )
                            response += f"📝 **Description**: {vision_response}\n"
                            # spinning end
                            next(sp, None)
                        else:
                            # spinning end
                            next(sp, None)

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
        return self.categories

if __name__ == "__main__":
    agent = SwarmUIAgent()