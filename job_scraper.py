"""
Internship opportunity scraper
- Renders pages with Playwright
- Extracts (link_text, href) pairs + visible body text from DOM — no raw HTML sent to LLM
- Batches 5 companies per Gemini call
- Outputs structured JSON with title, location, deadline, url, team

Setup:
    pip install playwright google-generativeai pandas
    playwright install chromium

    Set env var: GEMINI_API_KEY=<your key>
    Get a free key at: https://aistudio.google.com/app/apikey
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from playwright.async_api import async_playwright
import google.generativeai as genai
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME      = "gemini-2.5-flash-lite"
INPUT_CSV       = "input/companies.csv"   # columns: company, website, hr_website, 2nd_link

today = date.today()
today = today.strftime("%Y%m%d")
OUTPUT_JSON     = f"internships_{today}.json"

BATCH_SIZE      = 5      # companies per Gemini call
MIN_TEXT_CHARS  = 200    # if body text below this, page likely didn't render
PAGE_TIMEOUT_MS = 20_000
RATE_LIMIT_SEC  = 2      # between Gemini calls

INTERN_KEYWORDS = ["intern", "internship", "co-op", "coop", "summer", "placement", "graduate"]
DEADLINE_TERMS  = ["deadline", "apply by", "closes", "closing date", "due date", "applications due"]

# ── Gemini setup ──────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ── Prompt ────────────────────────────────────────────────────────────────────

BATCH_PROMPT = """
You are a job listing parser. Below is page data extracted from job board pages
for several companies. For EACH company, extract all internship or co-op listings.

The data for each company contains:
  - LINKS: anchor tag text paired with the resolved URL from the page
  - TEXT: visible body text (for finding deadlines, locations, teams)

Rules:
  - Use the LINKS section to find job titles and their URLs — prefer real hrefs over nulls
  - Use the TEXT section to find deadlines, locations, and team/department info
  - Only include listings that are internships, co-ops, or entry-level new grad roles
  - For "url": use the full resolved URL from LINKS if available; otherwise null
  - For "deadline": look for dates near "deadline", "apply by", "closes", "closing date" in TEXT; null if not found
  - Do not hallucinate URLs or deadlines

Return ONLY a JSON object (no markdown, no explanation) with company names as keys:
{{
  "Company A": [
    {{"title": "...", "location": "...", "deadline": "...", "url": "...", "team": "..."}}
  ],
  "Company B": []
}}
Use empty array [] if no internship listings found for a company.

---
{batch_data}
"""

# ── DOM extraction ────────────────────────────────────────────────────────────

EXTRACT_JS = """
() => {
    // 1. Collect all anchor tags with non-empty text and href
    const links = [];
    document.querySelectorAll('a[href]').forEach(a => {
        const text = a.innerText.trim().replace(/\\s+/g, ' ');
        const href = a.href;   // already resolved to absolute URL by browser
        if (text && href && !href.startsWith('javascript:')) {
            links.push([text, href]);
        }
    });

    // 2. Visible body text (collapse whitespace)
    const bodyText = (document.body.innerText || '')
        .replace(/[ \\t]{2,}/g, ' ')
        .replace(/\\n{3,}/g, '\\n\\n')
        .trim();

    return { links, bodyText };
}
"""

def filter_job_links(links: list[tuple], base_url: str) -> list[tuple]:
    """
    Keep links that are plausibly job listings:
    - Link text contains intern keywords, OR
    - Href contains job-board path patterns
    Drop nav/footer noise (very short text, social media, etc.)
    """
    job_path_patterns = re.compile(
        r'/(job|jobs|career|careers|position|opening|role|apply|intern|internship|requisition)',
        re.I
    )
    noise_domains = {"twitter.com", "linkedin.com", "facebook.com", "instagram.com",
                     "youtube.com", "t.co", "mailto:"}

    seen_hrefs = set()
    filtered = []

    for text, href in links:
        # deduplicate
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        # drop noise domains
        domain = urlparse(href).netloc
        if any(nd in domain for nd in noise_domains):
            continue

        # drop very short or very long link texts (likely icons or paragraphs)
        if len(text) < 4 or len(text) > 120:
            continue

        text_lower = text.lower()
        href_lower = href.lower()

        is_intern_text = any(kw in text_lower for kw in INTERN_KEYWORDS)
        is_job_path    = bool(job_path_patterns.search(href_lower))

        if is_intern_text or is_job_path:
            filtered.append((text, href))

    return filtered[:150]  # cap to avoid token bloat


def extract_deadline_context(body_text: str) -> str:
    """Pull lines from body text that mention deadlines — feed to LLM for parsing."""
    lines = body_text.split("\n")
    relevant = []
    for i, line in enumerate(lines):
        if any(term in line.lower() for term in DEADLINE_TERMS):
            # grab surrounding lines for context
            start = max(0, i - 1)
            end   = min(len(lines), i + 3)
            relevant.extend(lines[start:end])
    return "\n".join(relevant[:60])  # at most ~60 lines


async def get_page_data(page, url: str, company: str) -> dict | None:
    """Render page and extract structured (links, text) — no raw HTML."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
        await page.wait_for_timeout(2000)
    except Exception as e:
        print(f"    [warn] {company}: page load issue — {e}")

    try:
        data = await page.evaluate(EXTRACT_JS)
    except Exception as e:
        print(f"    [error] {company}: JS extraction failed — {e}")
        return None

    links     = data.get("links", [])
    body_text = data.get("bodyText", "")

    if len(body_text) < MIN_TEXT_CHARS:
        print(f"    [warn] {company}: very little text ({len(body_text)} chars) — page may not have rendered")

    job_links        = filter_job_links(links, url)
    deadline_context = extract_deadline_context(body_text)

    # Condensed body text for location/team context (not full text — just first ~3000 chars)
    short_body = body_text[:3000]

    print(f"    [extract] {company}: {len(job_links)} job links, {len(body_text)} body chars")
    return {
        "company":          company,
        "job_links":        job_links,         # [(text, href), ...]
        "deadline_context": deadline_context,  # lines mentioning deadlines
        "short_body":       short_body,        # first chunk for location/team hints
    }


# ── Batch Gemini call ─────────────────────────────────────────────────────────

def format_batch_data(batch: list[dict]) -> str:
    """Format extracted page data for a batch prompt."""
    parts = []
    for item in batch:
        company = item["company"]
        links_str = "\n".join(
            f"  [{text}] → {href}"
            for text, href in item["job_links"]
        ) or "  (no job links found)"

        parts.append(
            f"=== {company} ===\n"
            f"LINKS:\n{links_str}\n\n"
            f"TEXT (deadline context):\n{item['deadline_context'] or '(none)'}\n\n"
            f"TEXT (body excerpt):\n{item['short_body']}\n"
        )
    return "\n\n".join(parts)


def call_gemini_batch(batch: list[dict]) -> dict[str, list]:
    batch_data = format_batch_data(batch)
    prompt     = BATCH_PROMPT.format(batch_data=batch_data)

    try:
        response = model.generate_content(prompt)
        raw      = response.text
    except Exception as e:
        print(f"  [error] Gemini batch call failed: {e}")
        return {item["company"]: [] for item in batch}

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [warn] Could not parse Gemini response: {e}\n  Raw: {cleaned[:300]}")
        return {item["company"]: [] for item in batch}


# ── Core scrape logic ─────────────────────────────────────────────────────────

async def scrape_all(rows: list[dict]) -> list[dict]:
    page_data_map: dict[str, dict] = {}   # company → extracted data
    results:       list[dict]      = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        company_count = 0
        # ── Phase 1: render all pages ─────────────────────────────────────────
        for row in rows:
            company_count += 1
            if company_count <= 11:

                continue

            company    = str(row.get("company", "Unknown")).strip()
            hr_url     = str(row.get("website") or "").strip()
            backup_url = str(row.get("fallback")   or "").strip()
            
            print(f"\n{'─'*50}")
            print(f"  {company}")

            if not hr_url:
                print("  [skip] no hr_website")
                page_data_map[company] = None
                continue

            data = await get_page_data(page, hr_url, company)

            # If primary URL yielded no job links, try backup
            if (not data or not data["job_links"]) and backup_url:
                print(f"  [fallback] trying 2nd_link: {backup_url}")
                data = await get_page_data(page, backup_url, company)

            page_data_map[company] = data
            time.sleep(1)  # polite crawl delay
        await browser.close()

    # ── Phase 2: batch Gemini calls ───────────────────────────────────────────
    valid_items = [d for d in page_data_map.values() if d is not None]
    batches     = [valid_items[i:i+BATCH_SIZE] for i in range(0, len(valid_items), BATCH_SIZE)]

    print(f"\n  Calling Gemini in {len(batches)} batch(es) of up to {BATCH_SIZE}...")

    all_listings: dict[str, list] = {}
    for i, batch in enumerate(batches):
        companies_in_batch = [b["company"] for b in batch]
        print(f"  Batch {i+1}: {companies_in_batch}")
        batch_result = call_gemini_batch(batch)
        all_listings.update(batch_result)
        if i < len(batches) - 1:
            time.sleep(RATE_LIMIT_SEC)

    # ── Phase 3: assemble output ──────────────────────────────────────────────
    for row in rows:
        company = str(row.get("company", "Unknown")).strip()
        listings = all_listings.get(company, [])
        print(f"  {company}: {len(listings)} listing(s)")
        results.append({
            "company":    company,
            "source_url": str(row.get("hr_website") or ""),
            "listings":   listings,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    if not GEMINI_API_KEY:
        raise ValueError("Set the GEMINI_API_KEY environment variable.")

    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    rows = df.to_dict(orient="records")

    print(f"Loaded {len(rows)} companies from {INPUT_CSV}")
    results = await scrape_all(rows)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total             = sum(len(r["listings"]) for r in results)
    companies_with_hits = sum(1 for r in results if r["listings"])
    print(f"\n{'='*50}")
    print(f"  Done. {total} listing(s) across {companies_with_hits}/{len(results)} companies.")
    print(f"  Output → {OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())