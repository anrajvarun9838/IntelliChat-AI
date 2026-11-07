"""LangChain tools available in AI Agent mode.

All tools are async or fast, validate their input and return short text. None of
them executes model-generated code. Optional tools (Wikipedia, weather,
currency) use free, key-less public APIs and fail with a readable message when
the service is unreachable. `save_memory_fact` is the only tool that changes
stored data, so the graph asks the user to approve it first.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import database as db
from config import settings
from utils import safe_eval

HTTP_TIMEOUT = 12.0
HEADERS = {"User-Agent": "IntelliChatAI/1.0 (learning project)"}
MAX_TOOL_OUTPUT = 4000

# Tools whose execution needs the user's explicit approval (see graph.approval).
APPROVAL_TOOLS = {"save_memory_fact"}


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TOOL_OUTPUT else text[:MAX_TOOL_OUTPUT] + "\n…(truncated)"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Supports + - * / // % ** ^, parentheses,
    sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil, min, max, factorial, pi and e.
    Example: "(12.5 * 8) / 3 + sqrt(49)". Use this for ANY arithmetic instead of computing mentally."""
    try:
        return safe_eval(expression)
    except ValueError as exc:
        return f"Calculator error: {exc}"


@tool
def current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone such as "Asia/Kolkata", "America/New_York" or "UTC"."""
    from zoneinfo import ZoneInfo

    try:
        now = datetime.now(ZoneInfo(timezone_name))
    except Exception:  # unknown zone name or tzdata missing
        return f"Unknown timezone '{timezone_name}'. Use an IANA name like 'Asia/Kolkata'."
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z (UTC%z)")


@tool
async def web_search(query: str) -> str:
    """Search the web for current information, news or facts you do not know. Returns titles, URLs and snippets."""
    query = (query or "").strip()
    if not query or len(query) > 300:
        return "Provide a search query of 1-300 characters."
    try:
        if settings.tavily_api_key:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                    json={"query": query, "max_results": 5},
                )
                resp.raise_for_status()
                results = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                           for r in resp.json().get("results", [])]
        else:
            from ddgs import DDGS  # key-less fallback

            def _search() -> list[dict]:
                return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                        for r in DDGS().text(query, max_results=5)]

            results = await asyncio.to_thread(_search)
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model as text
        return f"Web search is unavailable right now ({type(exc).__name__})."
    if not results:
        return "No results found."
    return _clip("\n\n".join(f"{i}. {r['title']}\n{r['url']}\n{r['snippet'][:500]}"
                             for i, r in enumerate(results, 1)))


@tool
async def wikipedia_search(topic: str) -> str:
    """Get the introduction of the best-matching English Wikipedia article for a topic."""
    topic = (topic or "").strip()
    if not topic or len(topic) > 200:
        return "Provide a topic of 1-200 characters."
    params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": topic,
              "gsrlimit": 1, "prop": "extracts|info", "exintro": 1, "explaintext": 1,
              "exchars": 1500, "inprop": "url"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HEADERS) as client:
            resp = await client.get("https://en.wikipedia.org/w/api.php", params=params)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia is unavailable right now ({type(exc).__name__})."
    if not pages:
        return "No Wikipedia article found."
    page = next(iter(pages.values()))
    return _clip(f"{page.get('title')} ({page.get('fullurl', '')})\n{page.get('extract', '')}")


_WEATHER_CODES = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
                  51: "light drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
                  73: "snow", 75: "heavy snow", 80: "rain showers", 95: "thunderstorm"}


@tool
async def get_weather(city: str) -> str:
    """Get the current weather and a 3-day forecast for a city (temperatures in Celsius)."""
    city = (city or "").strip()
    if not city or len(city) > 100:
        return "Provide a city name."
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HEADERS) as client:
            geo = await client.get("https://geocoding-api.open-meteo.com/v1/search",
                                   params={"name": city, "count": 1})
            geo.raise_for_status()
            found = geo.json().get("results")
            if not found:
                return f"Could not find a place called '{city}'."
            place = found[0]
            forecast = await client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "forecast_days": 3})
            forecast.raise_for_status()
            data = forecast.json()
    except Exception as exc:  # noqa: BLE001
        return f"Weather service is unavailable right now ({type(exc).__name__})."
    cur, daily = data["current"], data["daily"]
    lines = [f"Weather in {place['name']}, {place.get('country', '')}: "
             f"{cur['temperature_2m']}°C, {_WEATHER_CODES.get(cur['weather_code'], 'unknown conditions')}, "
             f"humidity {cur['relative_humidity_2m']}%, wind {cur['wind_speed_10m']} km/h."]
    for day, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"],
                                 daily["temperature_2m_min"], daily["precipitation_sum"]):
        lines.append(f"{day}: {lo}°C to {hi}°C, precipitation {rain} mm")
    return "\n".join(lines)


@tool
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using the latest reference exchange rates (ECB data)."""
    src, dst = (from_currency or "").upper().strip(), (to_currency or "").upper().strip()
    if not (re.fullmatch(r"[A-Z]{3}", src) and re.fullmatch(r"[A-Z]{3}", dst)):
        return "Use 3-letter currency codes such as USD, EUR or INR."
    if not (0 <= amount <= 1e12):
        return "Amount must be between 0 and 1e12."
    if src == dst:
        return f"{amount} {src} = {amount} {dst}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HEADERS) as client:
            resp = await client.get("https://api.frankfurter.dev/v1/latest",
                                    params={"base": src, "symbols": dst})
            resp.raise_for_status()
            data = resp.json()
        rate = data["rates"][dst]
    except Exception as exc:  # noqa: BLE001
        return f"Currency service is unavailable or does not support that pair ({type(exc).__name__})."
    return f"{amount} {src} = {amount * rate:,.2f} {dst} (rate {rate}, date {data.get('date')})"


@tool
async def save_memory_fact(fact: str, config: RunnableConfig) -> str:
    """Save one short fact or preference about the user to their long-term memory.
    Only call this when the user explicitly asks you to remember something. The user must approve it."""
    fact = " ".join((fact or "").split())
    if not fact or len(fact) > 300:
        return "Provide a fact of 1-300 characters."
    user_id = (config or {}).get("configurable", {}).get("user_id")
    if not user_id:
        return "Cannot save memory: no user context."
    await asyncio.to_thread(db.add_memory, user_id, fact)
    return f"Saved to memory: {fact}"


TOOLS = [calculator, current_datetime, web_search, wikipedia_search, get_weather,
         convert_currency, save_memory_fact]
