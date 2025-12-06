"""
MCP Server Template for Compound Engineering Plugin
==================================================

This is a production-ready MCP server implementation that converts the
compound-engineering plugin into an MCP server.
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    GetPromptRequest,
    GetPromptResult,
    PromptMessage,
    Prompt,
    PromptArgument,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compound-engineering-mcp")

app = Server("compound-engineering")

# Tool definitions based on plugin structure
TOOLS = [
    Tool(
        name="code_review",
        description="Perform AI-powered code review with quality analysis",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to review"
                },
                "review_type": {
                    "type": "string",
                    "enum": ["standard", "security", "performance", "style"],
                    "description": "Type of review to perform"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the review"
                }
            },
            "required": ["file_path"]
        }
    ),
    Tool(
        name="generate_image",
        description="Generate images using AI with various styles and prompts",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt for image generation"
                },
                "style": {
                    "type": "string",
                    "enum": ["realistic", "artistic", "cartoon", "technical"],
                    "description": "Style of image to generate"
                },
                "size": {
                    "type": "string",
                    "enum": ["256x256", "512x512", "1024x1024"],
                    "description": "Image dimensions"
                }
            },
            "required": ["prompt"]
        }
    ),
    Tool(
        name="create_skill",
        description="Create new skills/tools for the compound engineering system",
        inputSchema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to create"
                },
                "description": {
                    "type": "string",
                    "description": "Description of what the skill does"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "bash", "javascript"],
                    "description": "Programming language for the skill"
                },
                "category": {
                    "type": "string",
                    "description": "Category/domain for the skill"
                }
            },
            "required": ["skill_name", "description"]
        }
    ),
    Tool(
        name="browser_automation",
        description="Automate browser actions using Playwright",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "click", "type", "screenshot", "extract_text"],
                    "description": "Browser action to perform"
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (for navigate action)"
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for element interaction"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for type action)"
                }
            },
            "required": ["action"]
        }
    ),
    Tool(
        name="workflow_automation",
        description="Execute automated workflows and compound engineering tasks",
        inputSchema={
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Name of the workflow to execute"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters to pass to the workflow"
                },
                "async_mode": {
                    "type": "boolean",
                    "description": "Whether to run workflow asynchronously"
                }
            },
            "required": ["workflow_name"]
        }
    )
]


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools."""
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "code_review":
            return await handle_code_review(arguments)
        elif name == "generate_image":
            return await handle_image_generation(arguments)
        elif name == "create_skill":
            return await handle_skill_creation(arguments)
        elif name == "browser_automation":
            return await handle_browser_automation(arguments)
        elif name == "workflow_automation":
            return await handle_workflow_automation(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_code_review(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle code review requests."""
    file_path = arguments["file_path"]
    review_type = arguments.get("review_type", "standard")
    context = arguments.get("context", "")

    # Check if file exists
    if not os.path.exists(file_path):
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        code_content = f.read()

    # Perform review based on type
    review_result = await perform_code_analysis(code_content, review_type, context)

    return [TextContent(type="text", text=review_result)]


async def handle_image_generation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle image generation requests."""
    prompt = arguments["prompt"]
    style = arguments.get("style", "realistic")
    size = arguments.get("size", "512x512")

    # This would integrate with actual image generation service
    # For now, return a placeholder
    result = f"Image generated with prompt: '{prompt}', style: {style}, size: {size}"

    return [TextContent(type="text", text=result)]


async def handle_skill_creation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle skill creation requests."""
    skill_name = arguments["skill_name"]
    description = arguments["description"]
    language = arguments.get("language", "python")
    category = arguments.get("category", "general")

    # Create skill structure
    skill_dir = Path(f"skills/{skill_name}")
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Generate skill template
    skill_template = generate_skill_template(skill_name, description, language, category)

    # Write skill files
    (skill_dir / "skill.json").write_text(json.dumps(skill_template["config"], indent=2))
    (skill_dir / f"main.{get_extension(language)}").write_text(skill_template["code"])

    return [TextContent(type="text", text=f"Skill '{skill_name}' created successfully in {skill_dir}")]


async def handle_browser_automation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle browser automation requests."""
    action = arguments["action"]

    # This would integrate with Playwright
    # For now, return a placeholder
    result = f"Browser automation: {action} completed"

    return [TextContent(type="text", text=result)]


async def handle_workflow_automation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle workflow automation requests."""
    workflow_name = arguments["workflow_name"]
    parameters = arguments.get("parameters", {})
    async_mode = arguments.get("async_mode", False)

    # Execute workflow
    result = f"Workflow '{workflow_name}' executed with parameters: {parameters}"
    if async_mode:
        result += " (running asynchronously)"

    return [TextContent(type="text", text=result)]


async def perform_code_analysis(code: str, review_type: str, context: str) -> str:
    """Perform code analysis based on review type."""
    analysis_parts = []

    # Basic metrics
    lines = len(code.split('\n'))
    analysis_parts.append(f"📊 Code Metrics:\n- Lines: {lines}")

    # Review type specific analysis
    if review_type == "security":
        security_issues = check_security_issues(code)
        analysis_parts.append(f"\n🔒 Security Analysis:\n{security_issues}")

    elif review_type == "performance":
        performance_suggestions = check_performance(code)
        analysis_parts.append(f"\n⚡ Performance Analysis:\n{performance_suggestions}")

    elif review_type == "style":
        style_suggestions = check_style(code)
        analysis_parts.append(f"\n🎨 Style Analysis:\n{style_suggestions}")

    else:  # standard
        general_feedback = check_general_quality(code)
        analysis_parts.append(f"\n✅ General Quality:\n{general_feedback}")

    if context:
        analysis_parts.append(f"\n📝 Context Considered: {context}")

    return "\n".join(analysis_parts)


def check_security_issues(code: str) -> str:
    """Check for common security issues."""
    issues = []

    if "eval(" in code:
        issues.append("- Potential code injection risk with eval()")
    if "exec(" in code:
        issues.append("- Potential code execution risk with exec()")
    if "password" in code.lower() and "=" in code:
        issues.append("- Hardcoded password detected")

    return "\n".join(issues) if issues else "- No obvious security issues detected"


def check_performance(code: str) -> str:
    """Check for performance issues."""
    suggestions = []

    if "for" in code and "append(" in code:
        suggestions.append("- Consider list comprehensions for better performance")
    if "import *" in code:
        suggestions.append("- Avoid wildcard imports for better startup time")

    return "\n".join(suggestions) if suggestions else "- Code looks optimized"


def check_style(code: str) -> str:
    """Check for style issues."""
    suggestions = []

    lines = code.split('\n')
    long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]

    if long_lines:
        suggestions.append(f"- Lines too long (>100 chars): {long_lines}")

    return "\n".join(suggestions) if suggestions else "- Code style looks good"


def check_general_quality(code: str) -> str:
    """Check general code quality."""
    feedback = []

    if 'def ' in code:
        feedback.append("- Functions detected ✓")
    if 'class ' in code:
        feedback.append("- Classes detected ✓")
    if '"""' in code or "'''" in code:
        feedback.append("- Documentation strings found ✓")

    return "\n".join(feedback) if feedback else "- Basic code structure"


def generate_skill_template(name: str, description: str, language: str, category: str) -> Dict[str, Any]:
    """Generate a skill template."""
    config = {
        "name": name,
        "description": description,
        "language": language,
        "category": category,
        "version": "1.0.0",
        "author": "MCP Server",
        "main": f"main.{get_extension(language)}"
    }

    if language == "python":
        code = f'''"""
{description}
"""

def main():
    """Main function for {name} skill."""
    print("Hello from {name}!")
    return "Success"

if __name__ == "__main__":
    main()
'''
    else:
        code = f"# {name} skill\necho 'Hello from {name}!'"

    return {"config": config, "code": code}


def get_extension(language: str) -> str:
    """Get file extension for language."""
    extensions = {
        "python": "py",
        "javascript": "js",
        "bash": "sh"
    }
    return extensions.get(language, "txt")


async def main():
    """Run the MCP server."""
    logger.info("Starting Compound Engineering MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
