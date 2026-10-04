from app.core.config import Settings
from app.tools.calculator import create_calculator_tool
from app.tools.current_time import create_current_time_tool
from app.tools.monid import create_monid_tools
from app.tools.registry import ToolRegistry
from app.tools.weather import create_weather_tool
from app.tools.web_search import create_web_search_tool


def create_default_tool_registry(settings: Settings) -> ToolRegistry:
    return ToolRegistry(
        tools=[
            create_calculator_tool(),
            create_current_time_tool(),
            create_weather_tool(),
            create_web_search_tool(),
            *create_monid_tools(settings),
        ]
    )
