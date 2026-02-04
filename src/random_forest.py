import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class FactorRandomForest:
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,n_jobs=-1)
        
        self.feature_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'Momentum_1M']
        self.target_col = 'Next_Month_Return'
        self.is_trained = False

    def prepare_data(self, merged_df):
        data = merged_df.copy()
        
        if 'date' in data.columns:
            data = data.sort_values(['Ticker', 'date'])
        else:
            data = data.sort_values(['Ticker'])

        data['Momentum_1M'] = data['Return']

        if 'Excess_Return' not in data.columns:
            data['Excess_Return'] = data['Return'] - data['RF']
            
        data[self.target_col] = data.groupby('Ticker')['Excess_Return'].shift(-1)
        data = data.dropna(subset=[self.target_col, 'Momentum_1M'])
        return data

    def train(self, train_df):
        X = train_df[self.feature_cols]
        y = train_df[self.target_col]
        
        print(f"Training Random Forest on {len(X)} rows with features: {self.feature_cols}")
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, test_df):
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call .train() first.")
            
        X_test = test_df[self.feature_cols]
        return self.model.predict(X_test)

    def get_feature_importance(self):
        if not self.is_trained:
            return None
            
        return pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)