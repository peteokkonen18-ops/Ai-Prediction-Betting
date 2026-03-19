#!/usr/bin/env python3
"""
World Prediction Core Telegram Bot

Features:
- Fetches today's football fixtures from API-Football
- Uses OpenAI GPT to generate football analysis
- Posts twice daily at configured times (comma-separated POST_TIME)
- Morning post: upcoming matches of the day
- Evening post: the most important match of the day, with deeper analysis
- Generates a match image using Pillow from template.png
- Sends the image to Telegram with the analysis caption

Install:
    pip install requests schedule openai pillow

Files needed:
    - template.png
    - A bold .ttf font file, e.g. DejaVuSans-Bold.ttf or arialbd.ttf

Run:
    python world_prediction_core_bot.py
"""

import html
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import schedule
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


# =========================
# CONFIGURATION VARIABLES
# =========================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")      
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID", "")      
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")          
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")         

TIMEZONE = "Europe/Helsinki"
POST_TIME = "08:00,20:00"     # Comma-separated list of times
OPENAI_MODEL = "gpt-4o-mini"
MAX_FIXTURES_PER_MORNING_POST = 8
REQUEST_TIMEOUT = 30

# Image settings
TEMPLATE_IMAGE_PATH = "template.png"
OUTPUT_IMAGE_PATH = "match_poster.png"
FONT_PATH = "DejaVuSans-Bold.ttf"   # Replace with your bold font file if needed
FONT_SIZE = 60
HOME_TEXT_Y = 420
AWAY_TEXT_Y = 630
TEXT_COLOR = "white"

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
TELEGRAM_API_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("world_prediction_core")


# =========================
# HELPERS
# =========================

def validate_config() -> None:
    missing = []

    if not TELEGRAM_BOT_TOKEN.strip():
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHANNEL_ID.strip():
        missing.append("TELEGRAM_CHANNEL_ID")
    if not OPENAI_API_KEY.strip():
        missing.append("OPENAI_API_KEY")
    if not API_FOOTBALL_KEY.strip():
        missing.append("API_FOOTBALL_KEY")

    if missing:
        raise ValueError(
            "Missing required config values: "
            + ", ".join(missing)
            + ". Please fill them at the top of the script."
        )

    if not os.path.exists(TEMPLATE_IMAGE_PATH):
        raise FileNotFoundError(
            f"Template image not found: {TEMPLATE_IMAGE_PATH}"
        )


def parse_post_times(post_time_value: str) -> List[str]:
    times = [t.strip() for t in post_time_value.split(",") if t.strip()]
    if not times:
        raise ValueError("POST_TIME must contain at least one time, e.g. '08:00,20:00'.")

    for t in times:
        try:
            time.strptime(t, "%H:%M")
        except ValueError as exc:
            raise ValueError(f"Invalid time format in POST_TIME: {t}. Use HH:MM.") from exc

    return times


def get_now_local() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def get_today_date_str() -> str:
    return get_now_local().strftime("%Y-%m-%d")


def make_request(method: str, url: str, **kwargs) -> requests.Response:
    response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def escape_for_telegram_html(text: str) -> str:
    allowed_tags = ["b", "/b", "i", "/i", "u", "/u", "s", "/s", "code", "/code", "pre", "/pre"]
    escaped = html.escape(text)
    for tag in allowed_tags:
        escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>")
    return escaped


def truncate_caption(text: str, max_len: int = 1024) -> str:
    """
    Telegram photo captions are limited.
    """
    text = escape_for_telegram_html(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# =========================
# API-FOOTBALL
# =========================

def fetch_today_fixtures() -> List[Dict[str, Any]]:
    today = get_today_date_str()
    url = f"{API_FOOTBALL_BASE_URL}/fixtures"

    headers = {
        "x-apisports-key": API_FOOTBALL_KEY,
    }

    params = {
        "date": today,
        "timezone": TIMEZONE,
    }

    logger.info("Fetching fixtures for %s (%s)", today, TIMEZONE)
    response = make_request("GET", url, headers=headers, params=params)
    data = response.json()
    fixtures = data.get("response", [])
    logger.info("Fetched %d fixtures", len(fixtures))
    return fixtures


def simplify_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    fixture_info = fixture.get("fixture", {})
    league = fixture.get("league", {})
    teams = fixture.get("teams", {})
    goals = fixture.get("goals", {})

    return {
        "fixture_id": fixture_info.get("id"),
        "date": fixture_info.get("date"),
        "timestamp": fixture_info.get("timestamp", 0),
        "status_short": fixture_info.get("status", {}).get("short", ""),
        "status_long": fixture_info.get("status", {}).get("long", ""),
        "league_name": league.get("name"),
        "league_country": league.get("country"),
        "league_round": league.get("round"),
        "league_id": league.get("id"),
        "home_team": teams.get("home", {}).get("name"),
        "away_team": teams.get("away", {}).get("name"),
        "home_winner": teams.get("home", {}).get("winner"),
        "away_winner": teams.get("away", {}).get("winner"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
    }


def get_upcoming_fixtures(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    upcoming_statuses = {"TBD", "NS", "1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT"}
    filtered = [fx for fx in fixtures if fx.get("status_short") in upcoming_statuses]
    return sorted(filtered, key=lambda x: x.get("timestamp", 0))


def score_fixture_importance(fixture: Dict[str, Any]) -> int:
    score = 0
    league_name = (fixture.get("league_name") or "").lower()
    round_name = (fixture.get("league_round") or "").lower()
    home_team = (fixture.get("home_team") or "").lower()
    away_team = (fixture.get("away_team") or "").lower()

    important_keywords = [
        "champions league",
        "europa league",
        "conference league",
        "premier league",
        "la liga",
        "serie a",
        "bundesliga",
        "ligue 1",
        "cup",
        "playoff",
        "play-offs",
        "final",
        "semi-final",
        "semifinal",
        "quarter-final",
        "derby",
    ]

    for keyword in important_keywords:
        if keyword in league_name:
            score += 20
        if keyword in round_name:
            score += 15

    big_clubs = [
        "real madrid", "barcelona", "atletico madrid",
        "manchester city", "manchester united", "liverpool", "arsenal", "chelsea",
        "bayern munich", "borussia dortmund",
        "juventus", "inter", "milan", "napoli",
        "psg", "paris saint germain",
        "benfica", "porto", "ajax", "celtic", "rangers"
    ]

    for club in big_clubs:
        if club in home_team:
            score += 10
        if club in away_team:
            score += 10

    if "final" in round_name:
        score += 30
    if "semi" in round_name:
        score += 20
    if "quarter" in round_name:
        score += 10

    return score


def pick_most_important_match(fixtures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    upcoming = get_upcoming_fixtures(fixtures)
    if not upcoming:
        return None

    ranked = sorted(
        upcoming,
        key=lambda x: (score_fixture_importance(x), x.get("timestamp", 0)),
        reverse=True
    )
    return ranked[0]


# =========================
# OPENAI PROMPTS
# =========================

def build_morning_prompt(fixtures: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, fx in enumerate(fixtures, start=1):
        lines.append(
            f"{idx}. {fx['home_team']} vs {fx['away_team']} | "
            f"League: {fx['league_name']} ({fx['league_country']}) | "
            f"Round: {fx['league_round']} | Kickoff: {fx['date']}"
        )

    fixtures_text = "\n".join(lines)

    return f"""
You are a football analyst writing for a Telegram channel called "World Prediction Core".

Create a morning post covering today's upcoming football matches.

Requirements:
- Write in English.
- Start with a strong headline for the morning post.
- Analyze each listed fixture briefly.
- For each match include:
  1. Match name
  2. Brief analysis (2-3 sentences)
  3. Prediction
  4. Suggested betting angle
  5. Confidence level out of 10
- Keep the post engaging and compact.
- Do not promise guaranteed wins.
- Use Telegram-friendly formatting.
- End with a short responsible betting disclaimer.

Today's upcoming fixtures:
{fixtures_text}
""".strip()


def build_evening_prompt(fixture: Dict[str, Any]) -> str:
    return f"""
You are a football analyst writing for a Telegram channel called "World Prediction Core".

Create an evening post focused on the single most important football match of the day.

Requirements:
- Write in English.
- Start with a strong evening headline.
- Focus only on this one match.
- Include:
  1. Match name
  2. Why this is the standout match of the day
  3. Detailed tactical/game-flow analysis
  4. Key strengths and possible weaknesses for both sides
  5. Main prediction
  6. Safer betting angle
  7. Higher-risk betting angle
  8. Confidence level out of 10
- Make it more detailed than the morning post.
- Keep it readable for Telegram.
- Do not promise guaranteed wins.
- End with a short responsible betting disclaimer.

Match:
{fixture['home_team']} vs {fixture['away_team']}
League: {fixture['league_name']} ({fixture['league_country']})
Round: {fixture['league_round']}
Kickoff: {fixture['date']}
""".strip()


def analyze_with_openai(prompt: str) -> str:
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0.7,
        max_output_tokens=1800,
    )

    text = getattr(response, "output_text", "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return text


# =========================
# IMAGE GENERATION
# =========================

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try configured font first, then common fallbacks.
    """
    font_candidates = [
        FONT_PATH,
        "arialbd.ttf",
        "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]

    for candidate in font_candidates:
        try:
            if candidate and os.path.exists(candidate):
                return ImageFont.truetype(candidate, size=size)
            # Also try direct system resolution
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue

    logger.warning("Could not load a bold TTF font. Falling back to default font.")
    return ImageFont.load_default()


def fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Reduce font size until text fits within max_width.
    """
    size = start_size
    while size >= 24:
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return font
        size -= 2

    return load_font(24)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    image_width: int,
    y_center: int,
    text: str,
    fill: str = "white",
    start_font_size: int = 60,
    max_text_width_ratio: float = 0.82,
) -> None:
    """
    Draw text centered horizontally and vertically around y_center.
    """
    max_width = int(image_width * max_text_width_ratio)
    font = fit_text_to_width(draw, text, max_width, start_font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_width - text_width) / 2
    y = y_center - (text_height / 2)

    draw.text((x, y), text, font=font, fill=fill)


def generate_match_image(home_team: str, away_team: str, output_path: str = OUTPUT_IMAGE_PATH) -> str:
    """
    Load template.png and write team names in the requested positions.
    """
    image = Image.open(TEMPLATE_IMAGE_PATH).convert("RGBA")
    draw = ImageDraw.Draw(image)

    image_width, _ = image.size

    draw_centered_text(
        draw=draw,
        image_width=image_width,
        y_center=HOME_TEXT_Y,
        text=home_team,
        fill=TEXT_COLOR,
        start_font_size=FONT_SIZE,
    )

    draw_centered_text(
        draw=draw,
        image_width=image_width,
        y_center=AWAY_TEXT_Y,
        text=away_team,
        fill=TEXT_COLOR,
        start_font_size=FONT_SIZE,
    )

    image.save(output_path)
    logger.info("Generated match image: %s", output_path)
    return output_path


# =========================
# TELEGRAM
# =========================

def send_telegram_message(message: str) -> Dict[str, Any]:
    url = f"{TELEGRAM_API_BASE_URL}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": escape_for_telegram_html(message),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    logger.info("Sending text message to Telegram channel %s", TELEGRAM_CHANNEL_ID)
    response = make_request("POST", url, json=payload)
    return response.json()


def send_telegram_photo(image_path: str, caption: str) -> Dict[str, Any]:
    url = f"{TELEGRAM_API_BASE_URL}/sendPhoto"

    with open(image_path, "rb") as photo_file:
        files = {
            "photo": photo_file,
        }
        data = {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "caption": truncate_caption(caption),
            "parse_mode": "HTML",
        }

        logger.info("Sending photo to Telegram channel %s", TELEGRAM_CHANNEL_ID)
        response = make_request("POST", url, data=data, files=files)
        return response.json()


# =========================
# POST JOBS
# =========================

def run_morning_post() -> None:
    try:
        logger.info("Starting morning post job")
        raw_fixtures = fetch_today_fixtures()
        fixtures = [simplify_fixture(fx) for fx in raw_fixtures]
        upcoming = get_upcoming_fixtures(fixtures)[:MAX_FIXTURES_PER_MORNING_POST]

        if not upcoming:
            message = (
                "<b>World Prediction Core — Morning Update</b>\n\n"
                "No upcoming matches were found for today."
            )
            send_telegram_message(message)
            logger.info("Morning text-only post sent successfully")
            return

        prompt = build_morning_prompt(upcoming)
        message = analyze_with_openai(prompt)

        # Use the first match from the morning selection for the image
        featured_match = upcoming[0]
        image_path = generate_match_image(
            home_team=featured_match["home_team"],
            away_team=featured_match["away_team"],
        )

        send_telegram_photo(image_path, message)
        logger.info("Morning photo post sent successfully")

    except Exception as exc:
        logger.exception("Morning post failed: %s", exc)


def run_evening_post() -> None:
    try:
        logger.info("Starting evening post job")
        raw_fixtures = fetch_today_fixtures()
        fixtures = [simplify_fixture(fx) for fx in raw_fixtures]
        top_match = pick_most_important_match(fixtures)

        if not top_match:
            message = (
                "<b>World Prediction Core — Evening Spotlight</b>\n\n"
                "No standout match was found for today."
            )
            send_telegram_message(message)
            logger.info("Evening text-only post sent successfully")
            return

        prompt = build_evening_prompt(top_match)
        message = analyze_with_openai(prompt)

        image_path = generate_match_image(
            home_team=top_match["home_team"],
            away_team=top_match["away_team"],
        )

        send_telegram_photo(image_path, message)
        logger.info("Evening photo post sent successfully")

    except Exception as exc:
        logger.exception("Evening post failed: %s", exc)


# =========================
# SCHEDULER
# =========================

running = True


def stop_gracefully(signum, frame):
    global running
    logger.info("Received signal %s. Shutting down...", signum)
    running = False


def schedule_jobs() -> None:
    times = parse_post_times(POST_TIME)

    for post_time in times:
        if post_time == "08:00":
            schedule.every().day.at(post_time).do(run_morning_post)
            logger.info("Scheduled morning post at %s (%s)", post_time, TIMEZONE)
        elif post_time == "20:00":
            schedule.every().day.at(post_time).do(run_evening_post)
            logger.info("Scheduled evening post at %s (%s)", post_time, TIMEZONE)
        else:
            schedule.every().day.at(post_time).do(run_morning_post)
            logger.info(
                "Scheduled generic daily post at %s (%s) using morning analysis mode",
                post_time,
                TIMEZONE,
            )


def main() -> None:
    validate_config()
    schedule_jobs()

    logger.info("Bot started")
    logger.info("Timezone: %s", TIMEZONE)
    logger.info("Configured post times: %s", POST_TIME)

    signal.signal(signal.SIGINT, stop_gracefully)
    signal.signal(signal.SIGTERM, stop_gracefully)

    while running:
        schedule.run_pending()
        time.sleep(20)

    logger.info("Bot stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)