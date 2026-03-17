#!/bin/bash

# MCP Server Wrapper Script for Apolo MCP Server

cd "$(dirname "$0")"
source .venv/bin/activate

python -m mcp_rag_server
