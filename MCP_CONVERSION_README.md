# MCP Server Configuration for Compound Engineering

This directory contains an MCP server implementation based on the compound-engineering-plugin.

## Quick Start

1. **Install Dependencies**
```bash
pip install mcp
```

2. **Run the Server**
```bash
python compound_engineering_mcp.py
```

3. **Configure in Claude Desktop**
Add to your Claude Desktop config (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "compound-engineering": {
      "command": "python",
      "args": ["/path/to/compound_engineering_mcp.py"],
      "env": {}
    }
  }
}
```

## Available Tools

### 1. Code Review
- **Purpose**: AI-powered code analysis and quality review
- **Usage**: Analyzes files for security, performance, style, or general quality
- **Input**: File path, review type, optional context

### 2. Image Generation  
- **Purpose**: Generate images using AI prompts
- **Usage**: Create images with various styles and specifications
- **Input**: Text prompt, style, size

### 3. Skill Creation
- **Purpose**: Create new skills/tools for the system
- **Usage**: Generates skill templates in Python, JavaScript, or Bash
- **Input**: Skill name, description, language, category

### 4. Browser Automation
- **Purpose**: Automate browser interactions
- **Usage**: Navigate, click, type, screenshot, extract text
- **Input**: Action type, URL, selectors, text

### 5. Workflow Automation
- **Purpose**: Execute complex automated workflows
- **Usage**: Run compound engineering tasks
- **Input**: Workflow name, parameters, async mode

## Architecture

```
compound_engineering_mcp.py
├── Tool Definitions (TOOLS)
├── Request Handlers
│   ├── handle_code_review()
│   ├── handle_image_generation()
│   ├── handle_skill_creation()
│   ├── handle_browser_automation()
│   └── handle_workflow_automation()
├── Analysis Functions
│   ├── perform_code_analysis()
│   ├── check_security_issues()
│   ├── check_performance()
│   └── check_style()
└── Utility Functions
```

## Plugin to MCP Mapping

| Plugin Component | MCP Equivalent | Description |
|------------------|----------------|-------------|
| `skills/` | Tools | Reusable functions/capabilities |
| `agents/` | Tool handlers | Business logic implementations |
| `commands/` | Tool definitions | Available operations |
| `.claude-plugin/plugin.json` | Server metadata | Configuration and capabilities |

## Extending the Server

### Adding New Tools

1. **Define Tool Schema**
```python
new_tool = Tool(
    name="my_tool",
    description="What my tool does", 
    inputSchema={
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        }
    }
)
TOOLS.append(new_tool)
```

2. **Implement Handler**
```python
async def handle_my_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    param = arguments["param"]
    result = do_something(param)
    return [TextContent(type="text", text=result)]
```

3. **Register in call_tool()**
```python
elif name == "my_tool":
    return await handle_my_tool(arguments)
```

### Plugin Integration Points

- **Skills Directory**: Map each skill to an MCP tool
- **Command Scripts**: Convert to async tool handlers
- **Agent Workflows**: Implement as complex tool chains
- **Configuration**: Merge plugin.json settings

## Development Tips

1. **Testing**: Use MCP Inspector for debugging
2. **Logging**: Enable detailed logging for troubleshooting  
3. **Error Handling**: Wrap tool calls in try-catch blocks
4. **Async**: Use async/await for I/O operations
5. **Validation**: Validate inputs against schemas

## Deployment Options

### Local Development
```bash
python compound_engineering_mcp.py
```

### Docker Container
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install mcp
CMD ["python", "compound_engineering_mcp.py"]
```

### HTTP Server Mode
```python
# Add HTTP server support
from mcp.server.fastapi import create_app
app = create_app(mcp_server)
```

## Security Considerations

- Validate all file paths to prevent directory traversal
- Sanitize shell commands to prevent injection
- Implement rate limiting for resource-intensive operations
- Use environment variables for sensitive configuration
- Restrict file system access to designated directories

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Claude Plugin Documentation](https://docs.anthropic.com/claude/docs/plugins)
- [Original Plugin Repository](https://github.com/EveryInc/compound-engineering-plugin)
