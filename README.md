# NBA-Predict: Predict Scores of NBA Games
This repo contains a machine-learning model that predicts the box scores of future NBA games based on data from seasons 2016-2024
1. Web Scraping
   - Uses BeautifulSoup and Playwright to scrap data from past NBA games
   - Gets the data from basketball-reference.com
2. Parses the data into desired format
   - Extract game statistics and create new features needed for machine learning later
   - Output nba_games.csv: a CSV file with all data parsed and ready
3. Machine Learning
   - Uses RidgeClassifier to make predictions
   - Trains the model with the data from the first few seasons in nba_games.csv
   - Checks accuracy by predicting box scores of games in the latter half of nba_games.csv
