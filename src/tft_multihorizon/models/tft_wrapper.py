class TFTWrapper:
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def fit(self, train_df, val_df=None):
        print(f"Fit TFT placeholder for horizon={self.horizon}")

    def predict(self, df):
        raise NotImplementedError("Connect to your chosen TFT library here.")
