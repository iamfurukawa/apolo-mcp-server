"""CRUD tools for MCP RAG Server."""

from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import Tool, TextContent
import json
import uuid
from datetime import datetime


def register_crud_tools(server: Server) -> None:
    """Register CRUD tools with the MCP server."""
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available CRUD tools."""
        return [
            Tool(
                name="create_document",
                description="Create a new document in the RAG system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Document content"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Document metadata (title, source, tags, etc.)"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name (optional, default: 'default')"
                        }
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="read_document",
                description="Read a document by ID from the RAG system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to retrieve"
                        }
                    },
                    "required": ["document_id"]
                }
            ),
            Tool(
                name="update_document",
                description="Update an existing document in the RAG system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "Updated document content"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Updated document metadata"
                        }
                    },
                    "required": ["document_id"]
                }
            ),
            Tool(
                name="delete_document",
                description="Delete a document by ID from the RAG system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to delete"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation flag for deletion"
                        }
                    },
                    "required": ["document_id", "confirm"]
                }
            ),
            Tool(
                name="list_documents",
                description="List all documents in a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name (optional, default: 'default')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of documents to skip"
                        }
                    }
                }
            ),
            Tool(
                name="search_documents",
                description="Search documents using semantic search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name (optional, default: 'default')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Metadata filters"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle CRUD tool calls."""
        
        if name == "create_document":
            return await create_document(arguments)
        elif name == "read_document":
            return await read_document(arguments)
        elif name == "update_document":
            return await update_document(arguments)
        elif name == "delete_document":
            return await delete_document(arguments)
        elif name == "list_documents":
            return await list_documents(arguments)
        elif name == "search_documents":
            return await search_documents(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")


# Mock storage for demonstration (replace with actual RAG implementation)
_documents: Dict[str, Dict[str, Any]] = {}
_collections: Dict[str, List[str]] = {"default": []}


async def create_document(args: Dict[str, Any]) -> List[TextContent]:
    """Create a new document."""
    content = args["content"]
    metadata = args.get("metadata", {})
    collection = args.get("collection", "default")
    
    document_id = str(uuid.uuid4())
    document = {
        "id": document_id,
        "content": content,
        "metadata": {
            **metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        },
        "collection": collection
    }
    
    _documents[document_id] = document
    
    if collection not in _collections:
        _collections[collection] = []
    _collections[collection].append(document_id)
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "document_id": document_id,
            "message": f"Document created in collection '{collection}'"
        }, indent=2)
    )]


async def read_document(args: Dict[str, Any]) -> List[TextContent]:
    """Read a document by ID."""
    document_id = args["document_id"]
    
    if document_id not in _documents:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Document '{document_id}' not found"
            }, indent=2)
        )]
    
    document = _documents[document_id]
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "document": document
        }, indent=2)
    )]


async def update_document(args: Dict[str, Any]) -> List[TextContent]:
    """Update an existing document."""
    document_id = args["document_id"]
    
    if document_id not in _documents:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Document '{document_id}' not found"
            }, indent=2)
        )]
    
    document = _documents[document_id]
    
    if "content" in args:
        document["content"] = args["content"]
    
    if "metadata" in args:
        document["metadata"].update(args["metadata"])
    
    document["metadata"]["updated_at"] = datetime.utcnow().isoformat()
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "document_id": document_id,
            "message": "Document updated successfully"
        }, indent=2)
    )]


async def delete_document(args: Dict[str, Any]) -> List[TextContent]:
    """Delete a document."""
    document_id = args["document_id"]
    confirm = args.get("confirm", False)
    
    if not confirm:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Deletion not confirmed. Set confirm=true to delete."
            }, indent=2)
        )]
    
    if document_id not in _documents:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Document '{document_id}' not found"
            }, indent=2)
        )]
    
    document = _documents[document_id]
    collection = document["collection"]
    
    del _documents[document_id]
    
    if collection in _collections and document_id in _collections[collection]:
        _collections[collection].remove(document_id)
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "document_id": document_id,
            "message": f"Document deleted from collection '{collection}'"
        }, indent=2)
    )]


async def list_documents(args: Dict[str, Any]) -> List[TextContent]:
    """List documents in a collection."""
    collection = args.get("collection", "default")
    limit = args.get("limit", 50)
    offset = args.get("offset", 0)
    
    if collection not in _collections:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "collection": collection,
                "documents": [],
                "total": 0
            }, indent=2)
        )]
    
    document_ids = _collections[collection][offset:offset + limit]
    documents = [_documents[doc_id] for doc_id in document_ids if doc_id in _documents]
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "collection": collection,
            "documents": documents,
            "total": len(_collections[collection]),
            "returned": len(documents)
        }, indent=2)
    )]


async def search_documents(args: Dict[str, Any]) -> List[TextContent]:
    """Search documents (mock implementation)."""
    query = args["query"].lower()
    collection = args.get("collection", "default")
    limit = args.get("limit", 10)
    filters = args.get("filters", {})
    
    if collection not in _collections:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "query": query,
                "collection": collection,
                "results": []
            }, indent=2)
        )]
    
    # Simple text search (replace with semantic search in real implementation)
    results = []
    for doc_id in _collections[collection]:
        if doc_id in _documents:
            doc = _documents[doc_id]
            
            # Apply filters
            filter_match = True
            for key, value in filters.items():
                if key not in doc["metadata"] or doc["metadata"][key] != value:
                    filter_match = False
                    break
            
            if filter_match and query in doc["content"].lower():
                results.append(doc)
                if len(results) >= limit:
                    break
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "query": query,
            "collection": collection,
            "results": results,
            "total_found": len(results)
        }, indent=2)
    )]
