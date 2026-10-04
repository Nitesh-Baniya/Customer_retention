from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.tools.registry import RegisteredTool


class CurrentTimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def current_utc_time(_: BaseModel) -> str:
    return json.dumps({"utc_time": datetime.now(UTC).isoformat()})


def create_current_time_tool() -> RegisteredTool:
    return RegisteredTool(
        name="current_utc_time",
        description="Return the current date and time in UTC.",
        input_model=CurrentTimeInput,
        handler=current_utc_time,
    )
