# SwarmUI_MCP_Agent
A simple prototype SwarmUI MCP server with accompanying local LangChain Ollama agent. The prototype only exposes the
SwarmUI /GenerateText2Image endpoint. It uses Node.js, Typescript, Langchain, @ModelContextProtocol, Axios, Express,
and Pydantic (though Pydantic is mostly unused in this implementation).

# SwarmUI MCP Server
The SwarmUI MCP Server prototype is declared in server.ts

The /GenerateText2Image and /GetNewSession SwarmUI API endpoints are exposed via generateImage() and getSessionId() methods, which
are described in the (MCP) Server capabilities config when instantiating the Server. The Server
capabilities config is used by the agent to know about the available tools' input and output schemas and
descriptions. The text descriptions for these tools are used by the llm to understand the tool. When
the server.js Node server is started, the SwarmUIServer is initialized and an Express server is created
with a POST endpoint at '/generate-image' that calls SwarmUIServer.generateImage, then the Express server
is started. An Axios instance is created to make HTTP requests to the local SwarmUI API and stored in the SwarmUIServer class.

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
