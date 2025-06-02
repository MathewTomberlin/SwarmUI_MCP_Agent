# SwarmUI_MCP_Agent
A simple prototype SwarmUI MCP server with accompanying local LangChain Ollama agent. The prototype only exposes the
SwarmUI /GenerateText2Image endpoint. It uses Python, Node.js, Typescript, Langchain, @ModelContextProtocol, Axios, Express,
and Pydantic (though Pydantic is mostly unused in this implementation).

# SwarmUI MCP Server
The SwarmUI MCP Server prototype is declared in server.ts.

The /GenerateText2Image and /GetNewSession SwarmUI API endpoints are exposed via generateImage() and getSessionId() methods, which
are described in the (MCP) Server capabilities config when instantiating the Server. The Server
capabilities config is used by the agent to know about the available tools' input and output schemas and
descriptions. The text descriptions for these tools are used by the llm to understand the tool. When
the server.js Node server is started, the SwarmUIServer is initialized and an Express server is created
with a POST endpoint at '/generate-image' that calls SwarmUIServer.generateImage, then the Express server
is started. An Axios instance is created to make HTTP requests to the local SwarmUI API and stored in the SwarmUIServer class.

DEV NOTE: You will likely need to change the port used by SwarmUI as well as the model hardcoded for image generation.

# GenerateImage StructuredTool
A generate_image StructuredTool is declared in tools.py that POSTS to the /generate-image endpoint of the Express server
started by the SwarmUIServer. The description of the StructuredTool and its docstring are used by the llm
to know about the tool. Theoretically, Pydantic models should assert tool input, but the current setup doesn't work for multiple
inputs yet.

# LangChain Ollama Agent
A local Ollama agent (currently a dolphin-mistral model) is instantiated with tool-calling instructions, a max_iterations of 1,
and AgentType of ZERO_SHOT_REACT_DESCRIPTION, with the generate_tool StructuredTool. User input is taken for the image
prompt and then the agent is invoked with the input. The agent will use the generate_image Action with the supplied prompt
as Action Input, calling the local SwarmUI API /GenerateText2Image endpoint with the prompt and other hardcoded image generation
settings.

# Starting the SwarmUI MCP Server
From the /SwarmUI_MCP_Agent/src directory, run: tsc server.ts && node server.js

# Running the Agent
With the SwarmUIMCPServer running and SwarmUI itself running, from the /SwarmUI_MCP_Agent directory, run: python run.py

# Viewing Generated Images
Generated images are available within the SwarmUI Image History or directly within the SwarmUI /Ouput directory

## Quickstart

### 1. Clone Repository
git clone <your-repo-url>
cd <your-repo-name>

### 2. Install Dependencies (Script)
Run setup.sh

### 2B. Install Dependencies (Manual)
npm install
pip install -r requirements.txt

### 3. Start the API server
tsc server.ts && node server.js

### 5. Start the UI
streamlit run run.py

### 6. Ensure SwarmUI is running and accessible.

### 7. Enter your SwarmUI Port number in the Agent Settings > SwarmUI API Port

### 8. Select LLM Model and (optionally) Vision Model

### 9. Enter image request and click submit. Wait for models to download, if not downloaded.
