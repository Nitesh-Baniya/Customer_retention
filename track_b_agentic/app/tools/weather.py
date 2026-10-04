from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import RegisteredTool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

logger = logging.getLogger(__name__)


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str = Field(
        min_length=2,
        max_length=100,
        description="City or place name, optionally followed by a country.",
    )


@dataclass(frozen=True, slots=True)
class ResolvedLocation:
    name: str
    country: str
    latitude: float
    longitude: float


class WeatherService:
    """Fetch current conditions from the free Open-Meteo API."""

    WEATHER_CODES: ClassVar[dict[int, str]] = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    async def current_weather(self, location_query: str) -> dict[str, object]:
        """Resolve a place and return its current weather conditions."""

        async with httpx.AsyncClient(
            timeout=20.0,
            headers={"User-Agent": "engineering-ai-assistant/0.1"},
        ) as client:
            for attempt in range(2):
                try:
                    # Step 1: Convert the human-readable place into coordinates.
                    location = await self._resolve_location(client, location_query)

                    # Step 2: Fetch current conditions for those coordinates.
                    current, units, timezone = await self._fetch_conditions(
                        client,
                        location,
                    )

                    # Step 3: Return a compact result suitable for the LLM context.
                    return self._build_result(location, current, units, timezone)
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Weather request failed on attempt %d: %s",
                        attempt + 1,
                        type(exc).__name__,
                    )
                    if attempt == 0:
                        await asyncio.sleep(0.5)

        raise ValueError("The weather service is currently unavailable")

    @staticmethod
    async def _resolve_location(
        client: httpx.AsyncClient,
        query: str,
    ) -> ResolvedLocation:
        response = await client.get(
            GEOCODING_URL,
            params={"name": query, "count": 1, "language": "en", "format": "json"},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise ValueError(f"Location not found: {query}")

        match = results[0]
        try:
            return ResolvedLocation(
                name=str(match["name"]),
                country=str(match.get("country", "")),
                latitude=float(match["latitude"]),
                longitude=float(match["longitude"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Geocoding service returned incomplete data") from exc

    @staticmethod
    async def _fetch_conditions(
        client: httpx.AsyncClient,
        location: ResolvedLocation,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        response = await client.get(
            FORECAST_URL,
            params={
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "weather_code,wind_speed_10m"
                ),
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current")
        if not isinstance(current, dict):
            raise ValueError("Weather service returned no current conditions")

        units = payload.get("current_units", {})
        return current, units, str(payload.get("timezone", "unknown"))

    def _build_result(
        self,
        location: ResolvedLocation,
        current: dict[str, Any],
        units: dict[str, Any],
        timezone: str,
    ) -> dict[str, object]:
        try:
            weather_code = int(current["weather_code"])
            return {
                "location": location.name,
                "country": location.country,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "timezone": timezone,
                "observed_at": current["time"],
                "condition": self.WEATHER_CODES.get(weather_code, "Unknown"),
                "weather_code": weather_code,
                "temperature": current["temperature_2m"],
                "temperature_unit": units.get("temperature_2m", "°C"),
                "apparent_temperature": current["apparent_temperature"],
                "relative_humidity_percent": current["relative_humidity_2m"],
                "wind_speed": current["wind_speed_10m"],
                "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
            }
        except (KeyError, TypeError) as exc:
            raise ValueError("Weather service returned incomplete data") from exc


async def get_current_weather(input_data: BaseModel) -> str:
    weather_input = WeatherInput.model_validate(input_data.model_dump())
    result = await WeatherService().current_weather(weather_input.location)
    return json.dumps(result)


def create_weather_tool() -> RegisteredTool:
    return RegisteredTool(
        name="get_current_weather",
        description="Get the current weather for a city or place.",
        input_model=WeatherInput,
        handler=get_current_weather,
    )
