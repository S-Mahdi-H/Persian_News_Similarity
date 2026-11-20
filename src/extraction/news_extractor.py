
import asyncio
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# News URLs
url_a = "https://www.irna.ir/news/86326529/"
url_b = "https://www.irna.ir/news/86326612/"


# Google Chrome path
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


async def extract_news(url):

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True
        )

        page = await browser.new_page(locale="fa-IR")

        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            await page.wait_for_timeout(7000)

            html = await page.content()

            soup = BeautifulSoup(html, "lxml")

            # Find article body
            article_body = soup.select_one(
                "article#item .item-body"
            )

            if article_body is None:
                raise ValueError(f"Article body not found: {url}")

            # Extract only paragraph texts
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in article_body.find_all("p")
                if p.get_text(" ", strip=True)
            ]

            return "\n".join(paragraphs)

        finally:
            await browser.close()


async def main():

    # Create raw data directory if it does not exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Extract news texts
    news_a = await extract_news(url_a)
    news_b = await extract_news(url_b)

    # Save extracted texts
    file_a = RAW_DATA_DIR / "news_a.txt"
    file_b = RAW_DATA_DIR / "news_b.txt"

    file_a.write_text(news_a, encoding="utf-8")
    file_b.write_text(news_b, encoding="utf-8")

    print("News extraction completed successfully.")

    print(f"News A: {len(news_a)} characters")
    print(f"News B: {len(news_b)} characters")

    print(f"\nSaved: {file_a}")
    print(f"Saved: {file_b}")


if __name__ == "__main__":
    asyncio.run(main())
