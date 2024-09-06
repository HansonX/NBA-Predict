import os
import time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Constants
SEASONS = list(range(2016, 2025))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")

# Helper function to fetch HTML content
async def fetch_html_content(url, selector, sleep=9, retries=12):
    """
    Fetch HTML content from a given URL using Playwright
    """
    html = None
    for attempt in range(1, retries + 1):
        time.sleep(sleep * attempt)
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(f"Fetching {url}: {await page.title()}")
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout Error on attempt {attempt} for {url}")
            continue
        else:
            break
    return html

# Function to scrape standings for a given season
async def scrape_season_data(season):
    """
    Scrape standings data for a given NBA season.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await fetch_html_content(url, "#content .filter")

    if not html:
        print(f"Failed to retrieve season data for {season}")
        return

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{link['href']}" for link in links]

    for page_url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, page_url.split("/")[-1])
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}, skipping.")
            continue
        
        page_html = await fetch_html_content(page_url, "#all_schedule")
        if page_html:
            with open(save_path, "w+", encoding='utf-8') as file:
                file.write(page_html)

# Function to scrape game data from standings files
async def scrape_game_data(standings_file):
    """
    Scrape game data from a standings file.
    """
    with open(standings_file, 'r', encoding='utf-8') as file:
        html = file.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    box_score_urls = [f"https://www.basketball-reference.com{link.get('href')}" for link in links if link.get('href') and "boxscore" in link.get('href')]

    for game_url in box_score_urls:
        save_path = os.path.join(SCORES_DIR, game_url.split("/")[-1])
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}, skipping.")
            continue

        game_html = await fetch_html_content(game_url, "#content")
        if game_html:
            with open(save_path, "w+", encoding='utf-8') as file:
                file.write(game_html)

# Main function to orchestrate scraping tasks
async def main():
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    os.makedirs(SCORES_DIR, exist_ok=True)

    # Scrape data for each season
    for season in SEASONS:
        await scrape_season_data(season)

    # Scrape game data from standings files
    standings_files = os.listdir(STANDINGS_DIR)
    for season in SEASONS:
        files_for_season = [file for file in standings_files if str(season) in file]
        for standings_file in files_for_season:
            await scrape_game_data(os.path.join(STANDINGS_DIR, standings_file))

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())