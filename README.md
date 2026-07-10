# NBA Rising Stars: an NBA player improvement/breakout prediction model 

Using NBA data from `nba_api`, player stats and information are aggregated to reflect a player's season. With this season data, including prior trajectory data from the season prior, the model predicts the likeliness of a player having a "breakout" season.
“Breaking out” reflects an improvement in player usage and efficiency using a weighted composite of stats (PIE, USG%, TS%, and game score).

XGBoost, a decision-tree model is trained on player stats and predicts the breakout score of next season.

Important note: Rookies are excluded from this model as it uses prior trajectory data to inform the predictions

<img width="662" height="413" alt="image" src="https://github.com/user-attachments/assets/8b1bc456-622c-4094-9e77-ad6af8717032" />


## Data
nba_api's base stats, advanced stats, and draft year + position stats for players. A row of data indicates a specific player's stats accross a given year. 
Players with exceptionally low playing time(less than 20 games or >10 minutes per game) are filtered out, along with rookies and players over 35
The feature matrix used to train the model (x_train) included 528 rows of player-seasons with 33 columns of stats

## Methodology

### Data Collection
Season stats were pulled from the NBA Stats API (`nba_api`) across 7 seasons 
(2018-19 through 2024-25), covering both basic box score stats and advanced 
metrics (PIE, USG%, TS%, EFG%, BPM, etc.) for every player in each season.

### Target Variable
The model predicts a composite **breakout score** — a weighted combination of 
year-over-year improvement across four metrics:

- **PIE** (Player Impact Estimate) — weight: 0.35
- **USG%** (Usage Rate) — weight: 0.25
- **TS%** (True Shooting %) — weight: 0.20
- **Game Score** — weight: 0.20

Each delta is z-scored before combining so all four metrics contribute on a 
comparable scale regardless of their raw units.

### Feature Engineering
Features are built exclusively from the *previous* season's stats to prevent 
data leakage. Key feature categories include:

- **Player context** — age, age squared (to capture the non-linear development 
curve), years in league, position, undrafted flag, and a binary breakout window 
flag (ages 22-25, years 1-4 in the league)
- **Volume/role**: minutes per game, games played, usage rate, low-minutes flag
- **Efficiency**: TS%, EFG%, PIE, efficiency gap (USG% minus PIE)
- **Production**: points, assists, rebounds, steals, blocks, turnovers per game
- **Prior trajectory**: year-over-year deltas from the season before the 
previous season, capturing whether a player was already trending up or down

### Train/Test Split
A temporal split is used rather than a random split — the model trains on 
transitions from 2018-19 through 2022-23 and is tested on the 2023-24 
transition. This replicates the real-world scenario of predicting a future 
season from past data, and avoids leaking future information into training.

### Modeling
Two models were evaluated:

- **Linear Regression** (baseline) — MAE: 0.54, R²: -0.015
- **XGBoost** (primary model) — MAE: 0.52, R²: 0.03

XGBoost outperforms the linear baseline by learning non-linear relationships 
and feature interactions that linear regression cannot capture — for example, 
the interaction between age and usage rate in identifying breakout candidates.

### Interpretability
SHAP (SHapley Additive exPlanations) values were used to understand which 
features drive individual predictions. The most influential features were:

1. Age
2. Prior delta in USG%
3. Prior delta in PIE
4. TS%
5. Years in league

## How to run it:
- Install dependencies:
   pip install nba_api xgboost shap pandas scikit-learn 
- Run pipeline.ipynb top to bottom 


## Known limitations and future improvements:
- A lot of what contributes to a player breakout is purely contextual, such as a change in team or coaching, which isn’t reflected in this model; pulling team data could help add context and complexity to the model
- NBA predictions can often be wildly inaccurate (we only achieved an r-squared of 0.04 when training the model), and injuries can vastly change a player’s career trajectory
- COVID seasons hinder data(shortened season and different environment, but per-game stats help mitigate this)
- Model only trained on 7 seasons, more data could show more nuanced patterns




