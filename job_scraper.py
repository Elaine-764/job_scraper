"""
Internship opportunity scraper
- Renders pages with Playwright
- Extracts visible text, passes to Gemini (free tier) for parsing
- Falls back to screenshot if text extraction yields too little content
- Outputs structured JSON

Setup:
    pip install playwright google-generativeai pandas pillow
    playwright install chromium
    
    Set env var: GEMINI_API_KEY=<your key>
    Get a free key at: https://aistudio.google.com/app/apikey
"""

# TODO: 
# - Batching: 1 call per 3-5 companies
# - try to fetch internship application deadlines/urls
# - avoid image prompts

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright
import google.generativeai as genai
from PIL import Image
import io

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-2.5-flash-lite"   # free tier; generous daily quota
INPUT_CSV      = "companies.csv"      # columns: company, website, backup
SCREENSHOT_DIR = Path("screenshots")

MIN_TEXT_CHARS  = 300    # below this → fall back to screenshot
PAGE_TIMEOUT_MS = 20_000
RATE_LIMIT_SEC  = 2      # polite delay between Gemini calls

KEYWORDS = ["intern", "internship", "co-op", "coop", "summer", "placement"]

# ── Gemini setup ──────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ── Prompts ───────────────────────────────────────────────────────────────────

TEXT_PROMPT = """
You are a job listing parser. Extract ALL internship or co-op opportunities 
from the following job board page text.

Return ONLY a JSON array. Each object must have:
  - "title": job title (string)
  - "location": location or "Remote" or "Not specified" (string)  
  - "deadline": application deadline if visible, else null (string or null)
  - "link": URL or path if visible, else null (string or null)
  - "team": team/department if visible, else null (string or null)

If no internship listings are found, return an empty array: []
Do not include any explanation, markdown, or code fences — raw JSON only.

Page text:
{page_text}
"""

IMAGE_PROMPT = """
You are a job listing parser. Extract ALL internship or co-op opportunities 
visible in this screenshot of a job board page.

Return ONLY a JSON array. Each object must have:
  - "title": job title (string)
  - "location": location or "Remote" or "Not specified" (string)
  - "deadline": application deadline if visible, else null (string or null)
  - "link": URL or path if visible, else null (string or null)
  - "team": team/department if visible, else null (string or null)

If no internship listings are found, return an empty array: []
Do not include any explanation, markdown, or code fences — raw JSON only.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON safely."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        print(f"    [warn] Could not parse JSON: {cleaned[:200]}")
        return []


def has_internship_keywords(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in KEYWORDS)


async def get_page_content(page, url: str) -> tuple[str, bytes | None]:
    """
    Navigate to URL and return (visible_text, screenshot_bytes).
    screenshot_bytes is None unless we need it.
    """
    try:
        await page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
        await page.wait_for_timeout(2000)  # let lazy-loaded content settle
    except Exception as e:
        print(f"    [warn] Page load issue: {e}")

    text = await page.inner_text("body")
    text = re.sub(r"\s{3,}", "\n\n", text).strip()  # collapse whitespace

    screenshot = None
    # if len(text) < MIN_TEXT_CHARS or not has_internship_keywords(text):
    screenshot = await page.screenshot(full_page=True) # change to always take screenshot

    return text, screenshot


def call_gemini_text(company: str, text: str) -> list[dict]:
    prompt = TEXT_PROMPT.format(page_text=text[:12000])  # stay within limits
    try:
        response = model.generate_content(prompt)
        return parse_json_response(response.text)
    except Exception as e:
        print(f"    [error] Gemini text call failed for {company}: {e}")
        return []


def call_gemini_image(company: str, screenshot_bytes: bytes) -> list[dict]:
    img = Image.open(io.BytesIO(screenshot_bytes))
    # Resize if very tall — Gemini has image size limits
    max_height = 4000
    if img.height > max_height:
        ratio = max_height / img.height
        img = img.resize((int(img.width * ratio), max_height))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_part = {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}

    try:
        response = model.generate_content([IMAGE_PROMPT, image_part])
        return parse_json_response(response.text)
    except Exception as e:
        print(f"    [error] Gemini image call failed for {company}: {e}")
        return []


def save_screenshot(company: str, url_label: str, data: bytes):
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w]", "_", company)
    path = SCREENSHOT_DIR / f"{safe}_{url_label}.png"
    path.write_bytes(data)
    print(f"    [screenshot] saved → {path}")
    return str(path)


# ── Core scrape logic ─────────────────────────────────────────────────────────

async def scrape_company(page, row: dict) -> dict:
    company    = row.get("company", "Unknown")
    hr_url     = str(row.get("website") or "").strip()
    backup_url = str(row.get("backup") or "").strip()

    print(f"\n{'─'*50}")
    print(f"  Company : {company}")
    print(f"  URL     : {hr_url or '(none)'}")

    results = {"company": company, "source_url": hr_url, "listings": [], "screenshot_path": None, "method": None}

    if not hr_url or len(hr_url.replace(' ', '')) == 0:
        print("  [skip] no website provided")
        return results

    urls_to_try = [u for u in [hr_url, backup_url] if u]

    for url in urls_to_try:
        print(f"  Fetching: {url}")
        text, screenshot = await get_page_content(page, url)

        # ── Path A: enough text and keywords present ──
        if len(text) >= MIN_TEXT_CHARS and has_internship_keywords(text):
            print(f"  [text]  {len(text)} chars — calling Gemini (text mode)")
            listings = call_gemini_text(company, text)
            results["method"] = "text"

            if listings:
                results["listings"] = listings
                results["source_url"] = url
                print(f"  [ok]    {len(listings)} listing(s) found")
                return results
            else:
                print("  [info]  Gemini found 0 listings via text — trying screenshot")
                screenshot = await page.screenshot(full_page=True)

        # ── Path B: screenshot fallback ──
        if screenshot:
            label = "primary" if url == hr_url else "secondary"
            results["screenshot_path"] = save_screenshot(company, label, screenshot)
            print(f"  [image] calling Gemini (screenshot mode)")
            listings = call_gemini_image(company, screenshot)
            results["method"] = "screenshot"

            if listings:
                results["listings"] = listings
                results["source_url"] = url
                print(f"  [ok]    {len(listings)} listing(s) found via screenshot")
                return results
            else:
                print(f"  [info]  0 listings found on {url} — trying next URL if available")

        time.sleep(RATE_LIMIT_SEC)

    print(f"  [done]  total listings: {len(results['listings'])}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    if not GEMINI_API_KEY:
        raise ValueError("Set the GEMINI_API_KEY environment variable.")

    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()

    all_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        for _, row in df.iterrows():
            result = await scrape_company(page, row.to_dict())
            all_results.append(result)
            time.sleep(RATE_LIMIT_SEC)

        await browser.close()

    # ── Write output ──────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(len(r["listings"]) for r in all_results)
    companies_with_hits = sum(1 for r in all_results if r["listings"])
    print(f"\n{'='*50}")
    print(f"  Done. {total} listing(s) across {companies_with_hits}/{len(all_results)} companies.")
    print(f"  Output → {OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())