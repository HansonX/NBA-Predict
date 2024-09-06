import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load and prepare the dataset
def load_and_prepare_data(filepath):
    """
    Load the NBA games dataset, sort by date, and reset index.
    """
    df = pd.read_csv(filepath)
    df = df.sort_values("date").reset_index(drop=True)
    
    df.drop(columns=["mp.1", "mp_opp.1", "index_opp"], inplace=True)
    
    return df

# Add target column to the dataset
def add_target_column(df):
    """
    Add a target column for predicting the next game result.
    """
    df = df.groupby("team", group_keys=False).apply(lambda team: team.assign(target=team["won"].shift(-1)))
    df["target"].fillna(2, inplace=True)
    df["target"] = df["target"].astype(int, errors="ignore")
    return df

# Clean dataset by removing columns with missing values
def clean_dataset(df):
    """
    Remove columns with missing values from the dataset.
    """
    null_columns = df.columns[df.isnull().any()]
    return df.drop(columns=null_columns)

# Scale selected columns in the dataset
def scale_features(df, columns):
    """
    Scale the selected features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Perform feature selection using Sequential Feature Selector
def perform_feature_selection(df, model, split, removed_cols):
    """
    Perform feature selection using Sequential Feature Selector.
    """
    selected_cols = df.columns[~df.columns.isin(removed_cols)]
    sfs = SequentialFeatureSelector(model, n_features_to_select=30, direction="forward", cv=split, n_jobs=1)
    sfs.fit(df[selected_cols], df["target"])
    return list(selected_cols[sfs.get_support()])

# Backtest the model's predictions
def backtest(data, model, predictors, start=2):
    """
    Backtest the model by training and predicting on each season.
    """
    all_predictions = []
    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons)):
        train = data[data["season"] < seasons[i]]
        test = data[data["season"] == seasons[i]]

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        combined = pd.concat([test["target"], pd.Series(preds, index=test.index)], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)

    return pd.concat(all_predictions)

# Compute rolling averages for the dataset
def compute_rolling_averages(df, columns):
    """
    Compute rolling averages for the specified columns.
    """
    df_rolling = df.groupby(["team", "season"], group_keys=False).apply(lambda team: team[columns].rolling(10).mean())
    df_rolling.columns = [f"{col}_10" for col in df_rolling.columns]
    return df_rolling

# Shift specified column to the next game for each team
def shift_column(df, col_name):
    """
    Shift the specified column to the next game for each team.
    """
    return df.groupby("team", group_keys=False).apply(lambda team: team[col_name].shift(-1))

# Main function to run the analysis
def main():
    # Load and prepare data
    df = load_and_prepare_data("nba_games.csv")
    df = add_target_column(df)
    df = clean_dataset(df)

    # Define removed columns and perform feature selection
    removed_cols = ["season", "date", "won", "target", "team", "team_opp"]
    rr = RidgeClassifier(alpha=1)
    split = TimeSeriesSplit(n_splits=3)

    # Scale features
    selected_cols = df.columns[~df.columns.isin(removed_cols)]
    df = scale_features(df, selected_cols)

    # Compute rolling averages
    df_rolling = compute_rolling_averages(df, list(selected_cols) + ["won", "team", "season"])
    df = pd.concat([df, df_rolling], axis=1).dropna()

    # Add shifted columns for next game predictions
    df["home_next"] = shift_column(df, "home")
    df["team_opp_next"] = shift_column(df, "team_opp")
    df["date_next"] = shift_column(df, "date")

    # Merge datasets for full backtesting
    full = df.merge(df[df_rolling.columns.tolist() + ["team_opp_next", "date_next", "team"]],
                    left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

    # Update removed columns and perform feature selection again
    removed_cols.extend(full.columns[full.dtypes == "object"].tolist())
    selected_cols = full.columns[~full.columns.isin(removed_cols)]
    predictors = perform_feature_selection(full, rr, split, removed_cols)

    # Backtest and compute final accuracy
    predictions = backtest(full, rr, predictors)
    print(f"Final Accuracy: {accuracy_score(predictions['actual'], predictions['prediction'])}")

# Run the main function
if __name__ == "__main__":
    main()
