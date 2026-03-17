"""MCP RAG Server."""

from fastmcp import FastMCP
from .tools.rag_tools import (
    create_document,
    search_documents,
    get_document_by_id
)


# Initialize the server with a name
mcp = FastMCP("apolo-mcp-server")

# Register tools directly
mcp.add_tool(create_document)
mcp.add_tool(search_documents)
mcp.add_tool(get_document_by_id)

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
