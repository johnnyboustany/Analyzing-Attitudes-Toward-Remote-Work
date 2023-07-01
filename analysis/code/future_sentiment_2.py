import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and preprocess data
sent_path = '..\data\combined_scored_sentiment_ratings.json'
art_path = '..\..\data_deliverable\data_files\combined_articles.json'
sents = pd.read_json(sent_path)
arts = pd.read_json(art_path)
df = pd.merge(sents, arts, how='inner', on='ID').reset_index()
df = df[df['METHOD'] == 'roberta_average_sentence_sentiment_conf_scaled']
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[['ID', 'DATE', 'RATING']]
train_df = df[df['DATE'].dt.year.isin([2020, 2021])].iloc[2:]
test_df = df[df['DATE'].dt.year.isin([2022, 2023])]

# Define features (X) and target (y)
X = train_df[['RATING']]
y = test_df[['RATING']]

print(X.shape)
print(y.shape)
# Define number of folds
k = 5

# Initialize k-fold cross-validation
kf = KFold(n_splits=k)

# Initialize lists to store scores
r2_scores = []
mse_scores = []

# Loop through each fold
for train_idx, test_idx in kf.split(X):

    # Split data into training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Initialize model and fit to training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate R^2 and mean squared error
    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)

    # Append scores to lists
    r2_scores.append(r2)
    mse_scores.append(mse)

# Calculate mean scores across all folds
mean_r2 = np.mean(r2_scores)
mean_mse = np.mean(mse_scores)

output = {
    "mean_r2": mean_r2,
    "mean_mse": mean_mse,
}

with open('regression_output_2.json', 'w') as f:
    for key in output:
        f.write(f"{key}: {output[key]}\n")

# Print results
print(f"Mean R^2: {mean_r2}")
print(f"Mean MSE: {mean_mse}")
