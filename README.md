# AI Internship & Job Scraper

This project automates the process of finding internship and early-career opportunities across company career pages and job boards. It uses browser automation and LLM-based extraction to turn messy, unstructured web content into clean, structured job records.

## What this project does

- Visits company websites and career pages
- Extracts visible job-related links and page text
- Filters for internship, co-op, and entry-level roles
- Uses Gemini to parse the content into structured fields
- Saves the results as JSON and CSV files for review and analysis

## Why it matters

Job hunting is often a manual, repetitive process. This project reduces that effort by automating discovery and data extraction, helping users quickly identify relevant opportunities without reading every page manually.

## Tech stack

- Python
- Playwright for browser automation
- Google Generative AI (Gemini)
- pandas for data shaping and export
- JSON/CSV pipelines for downstream analysis

## Project flow

1. Reads a list of companies from a CSV file
2. Opens each company website in a browser
3. Extracts visible links and text from the page
4. Filters likely internship/job pages using URL and text heuristics
5. Sends grouped page data to Gemini for structured extraction
6. Normalizes results into fields like:
   - title
   - company
   - location
   - deadline
   - url
   - team
7. Saves outputs to JSON and CSV files for review

## Repository structure

- `job_scraper.py` — main scraper workflow using Playwright + Gemini
- `collect_jobs.py` — combines JSON role files into a single CSV
- `combine_roles.py` — normalizes and merges role records
- `companies.csv` — source list of companies and URLs
- `roles/` — generated job listing JSON outputs
- `combined_roles.csv` — aggregated final dataset

## Example output

Each role is stored as a structured record such as:

```json
{
  "title": "Software Engineering Intern",
  "company": "Example Corp",
  "location": "New York, NY",
  "deadline": "2026-10-15",
  "url": "https://example.com/jobs/123",
  "team": "Platform Engineering"
}
```

## Setup

```bash
pip install playwright google-generativeai pandas
playwright install chromium
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Then run the scraper:

```bash
python job_scraper.py
```

## Impact

This project demonstrates a practical application of automation, data cleaning, and AI-powered extraction to solve a real-world workflow problem. It combines software engineering, web scraping, and data processing into a useful end-to-end system.

## Project highlights

- Automated web extraction at scale
- LLM-assisted data structuring
- Clean dataset generation for job search workflows
- Useful for internship tracking, recruiting pipelines, and market analysis
