import os
import pandas as pd
from bs4 import BeautifulSoup

# Constants
SCORE_DIR = "data/scores"
BOX_SCORES = [os.path.join(SCORE_DIR, f) for f in os.listdir(SCORE_DIR) if f.endswith(".html")]

# Function to parse HTML content of a box score
def parse_html_content(file_path):
    """
    Parse the HTML content of a box score
    """
    with open(file_path, encoding='utf-8') as file:
        html = file.read()
    soup = BeautifulSoup(html, 'html.parser')
    for s in soup.select("tr.over_header, tr.thead"):
        s.decompose()
    return soup

# Function to read the line score table
def extract_line_score(soup):
    """
    Extract the line score from the parsed HTML.
    """
    line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0]
    cols = list(line_score.columns)
    cols[0], cols[-1] = "team", "total"
    line_score.columns = cols
    return line_score[["team", "total"]]

# Function to read team statistics
def extract_team_stats(soup, team, stat_type):
    """
    Extract team statistics from the parsed HTML.
    """
    df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat_type}"}, index_col=0)[0]
    return df.apply(pd.to_numeric, errors="coerce")

# Function to extract the season from the HTML content
def extract_season(soup):
    """
    Extract the season information from the parsed HTML.
    """
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

# Function to process a single box score file
def process_box_score(box_score, base_columns):
    """
    Process a single box score HTML file.
    """
    soup = parse_html_content(box_score)
    line_score = extract_line_score(soup)
    teams = list(line_score["team"])
    
    summaries = []
    for team in teams:
        basic_stats = extract_team_stats(soup, team, "basic")
        advanced_stats = extract_team_stats(soup, team, "advanced")
        
        totals = pd.concat([basic_stats.iloc[-1, :], advanced_stats.iloc[-1, :]])
        totals.index = totals.index.str.lower()

        maxes = pd.concat([basic_stats.iloc[:-1, :].max(), advanced_stats.iloc[:-1, :].max()])
        maxes.index = maxes.index.str.lower() + "_max"

        summary = pd.concat([totals, maxes])
        
        if not base_columns:
            base_columns = list(summary.index.drop_duplicates(keep="first"))
            base_columns = [col for col in base_columns if "bpm" not in col]
        
        summary = summary[base_columns]
        summaries.append(summary)
    
    summary_df = pd.concat(summaries, axis=1).T
    game_df = pd.concat([summary_df, line_score], axis=1)
    game_df["home"] = [0, 1]

    game_opp_df = game_df.iloc[::-1].reset_index(drop=True)
    game_opp_df.columns += "_opp"

    full_game_df = pd.concat([game_df, game_opp_df], axis=1)
    full_game_df["season"] = extract_season(soup)
    full_game_df["date"] = pd.to_datetime(os.path.basename(box_score)[:8], format="%Y%m%d")
    full_game_df["won"] = full_game_df["total"] > full_game_df["total_opp"]

    return full_game_df

# Main function to process all box score files
def main():
    base_columns = None
    games_data = []

    for i, box_score in enumerate(BOX_SCORES):
        game_data = process_box_score(box_score, base_columns)
        games_data.append(game_data)
        base_columns = game_data.columns if not base_columns else base_columns

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(BOX_SCORES)} files")

    games_df = pd.concat(games_data, ignore_index=True)
    games_df.to_csv('nba_games.csv', index=False)
    print("Data saved to 'nba_games.csv'.")

# Entry point
if __name__ == "__main__":
    main()