#!/bin/bash

# MCP Server Wrapper Script for Apolo MCP Server

# Set working directory
cd "/Users/iamfurukawa/Documents/Projects/apolo-mcp-server"

# Set Python path for imports
export PYTHONPATH="/Users/iamfurukawa/Documents/Projects/apolo-mcp-server/src"

# Activate virtual environment
source .venv/bin/activate

# Run the MCP server as module
python -m src.mcp_rag_server.server
