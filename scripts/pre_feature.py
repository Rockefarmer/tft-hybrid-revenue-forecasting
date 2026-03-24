import os
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class IntegratedTFTFeaturePipeline:
    """
    Unified pipeline merging:
    1. Preprocessing (Scaling, Date anchoring)
    2. Feature Engineering (Ratios, YoY, 12-period Lags)
    3. Re-Feature (Log transforms, Winsorizing, Time Indexes)
    """
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 tickers_csv_path: str):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuration Constants
        self.ANCHOR_SHIFT_DAYS = 5
        self._QMAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        self.WINSOR_LO = 0.01
        self.WINSOR_HI = 0.99
        self.USE_LOG_ASSETS = True
        self.DROP_INCOMPLETE_ENCODER_ROWS = True
        self.MINIMAL_ENCODER_COLS = ["revenue_log", "revenue_lag1"]
        
        # Load Universe
        self.tickers = self._read_universe(tickers_csv_path)
        print(f"Initialized Pipeline. Universe: {len(self.tickers)} tickers.")

    # ==========================================
    # 1. SETUP & UTILS
    # ==========================================
    def _read_universe(self, csv_path: str) -> list[str]:
        dfu = pd.read_csv(csv_path)
        col = next((c for c in ['symbol', 'Symbol', 'ticker', 'Ticker'] if c in dfu.columns), None)
        if not col: raise ValueError(f"No ticker column found in {csv_path}")
        return sorted(dfu[col].astype(str).str.strip().str.upper().dropna().unique().tolist())

    def _safe_div(self, numer: pd.Series, denom: pd.Series) -> pd.Series:
        return (numer / denom.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan)

    def _safe_log1p(self, x):
        return np.log1p(np.clip(x, a_min=0, a_max=None))

    def _winsorize_series(self, s: pd.Series):
        s_nonan = s.dropna()
        if s_nonan.empty: return s
        lo_v, hi_v = s_nonan.quantile(self.WINSOR_LO), s_nonan.quantile(self.WINSOR_HI)
        if not np.isfinite(lo_v) or not np.isfinite(hi_v): return s
        return s.clip(lower=lo_v, upper=hi_v)

    def avg_qoq_growth(self, s: pd.Series, n: int = 4) -> float:
        if s is None or s.empty: return 0.0
        growths = s.astype(float).pct_change().dropna().tail(n)
        return float(growths.mean()) if not growths.empty else 0.0

    def _next_year_quarter(self, y: int, q_int: int) -> tuple[int, str, int]:
        qn = q_int + 1
        yn = y + 1 if qn == 5 else y
        qn = 1 if qn == 5 else qn
        return yn, f"Q{qn}", qn

    # ==========================================
    # 2. CORE PROCESSING LOGIC
    # ==========================================
    def process_raw_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Stage 1: Clean raw FMP data and shift anchors."""
        df['date'] = pd.to_datetime(df['date'])
        anchor = df['date'] - pd.Timedelta(days=self.ANCHOR_SHIFT_DAYS)

        df['year'] = anchor.dt.year
        df['quarter'] = anchor.dt.quarter.apply(lambda x: f"Q{x}")
        df['quarter_int'] = anchor.dt.quarter
        df['Year_Quarter'] = df['year'].astype(str) + df['quarter']

        # Downscale to millions
        dollar_cols = [
            'revenue', 'costOfRevenue', 'grossProfit', 'researchAndDevelopmentExpenses', 
            'sellingGeneralAndAdministrativeExpenses', 'operatingExpenses', 'ebitda', 
            'operatingIncome', 'incomeBeforeTax', 'netIncome', 'totalAssets', 'totalEquity'
        ]
        for col in dollar_cols:
            if col in df.columns:
                df[col] = df[col] / 1_000_000

        df = df.rename(columns={'researchAndDevelopmentExpenses': 'rnd', 'sellingGeneralAndAdministrativeExpenses': 'snaExpenses'})
        
        # Sort chronologically to prep for lagging
        return df.sort_values(["year", "quarter_int"]).reset_index(drop=True)

    def engineer_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Calculate ratios, YoY, and long-horizon lags."""
        # 1. Ratios
        if {"rnd", "revenue"}.issubset(df.columns): df["rnd_to_rev_ratio"] = self._safe_div(df["rnd"], df["revenue"])
        if {"snaExpenses", "revenue"}.issubset(df.columns): df["sna_to_rev_ratio"] = self._safe_div(df["snaExpenses"], df["revenue"])
        if {"grossProfit", "revenue"}.issubset(df.columns): df["grossProfitRatio"] = self._safe_div(df["grossProfit"], df["revenue"])
        if {"operatingIncome", "revenue"}.issubset(df.columns): df["operatingIncomeRatio"] = self._safe_div(df["operatingIncome"], df["revenue"])
        if {"netIncome", "revenue"}.issubset(df.columns): df["netIncomeRatio"] = self._safe_div(df["netIncome"], df["revenue"])
        
        if {"netIncome", "rnd"}.issubset(df.columns): df["net_plus_rnd"] = df["netIncome"] + df["rnd"]

        # 2. Add 12-period Lags
        for col in ["rnd", "rnd_to_rev_ratio"]:
            if col in df.columns:
                for k in range(1, 13): df[f"{col}_lag{k}"] = df[col].shift(k)

        # Assets & Equity Lag 1 proxy
        if "totalAssets" in df.columns: df["totalAssets_lag1"] = df["totalAssets"].shift(1)
        if "totalEquity" in df.columns: df["totalEquity_lag1"] = df["totalEquity"].shift(1)

        # 3. Seasonal One-Hot
        q_int = df["quarter"].map(self._QMAP)
        for i in [1, 2, 3, 4]: df[f"Q{i}"] = (q_int == i).astype(int)

        # 4. YoY Growth
        yoy_targets = [c for c in ["revenue", "netIncome", "grossProfit", "rnd", "ebitda", "totalAssets"] if c in df.columns]
        for c in yoy_targets: df[f"{c}_yoy"] = self._safe_div(df[c], df[c].shift(4)) - 1.0

        return df

    def append_future_quarters(self, df: pd.DataFrame, n_future: int = 4) -> pd.DataFrame:
        """Stage 3: Append decoder horizons and strictly roll forward known lags."""
        if df.empty: return df

        last_hist = df.iloc[-1]
        last_year, last_qint = int(last_hist["year"]), int(self._QMAP[last_hist["quarter"]])
        
        ta_hist = df["totalAssets"] if "totalAssets" in df.columns else None
        ta_avg_g = self.avg_qoq_growth(ta_hist)
        ta_future = float(ta_hist.iloc[-1]) if ta_hist is not None else np.nan

        frames = [df]
        for h in range(1, n_future + 1):
            last_year, qstr, qint = self._next_year_quarter(last_year, last_qint)
            last_qint = qint

            row = {k: np.nan for k in df.columns}
            row["year"], row["quarter"], row["quarter_int"] = last_year, qstr, qint
            row["Year_Quarter"] = f"{last_year}{qstr}"
            if "ticker" in df.columns: row["ticker"] = last_hist["ticker"]

            tmp = pd.DataFrame([row])
            
            # One-Hots
            for i in [1, 2, 3, 4]: tmp[f"Q{i}"] = 1 if qint == i else 0

            # Roll Assets Forward
            if "totalAssets_lag1" in df.columns:
                if h > 1 and pd.notna(ta_future): ta_future *= (1.0 + ta_avg_g)
                tmp["totalAssets_lag1"] = ta_future

            frames.append(tmp[df.columns])
            
        return pd.concat(frames, ignore_index=True)

    def finalize_tft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 4: Target transforms, winsorizing, and time indices."""
        # 1. Target log transforms
        if "revenue" in df.columns: df["revenue_log"] = self._safe_log1p(df["revenue"])
        
        # 2. Revenue Lags (Computed AFTER future rows appended so future gets NaNs correctly)
        df["revenue_lag1"] = df["revenue"].shift(1)
        df["revenue_lag4"] = df["revenue"].shift(4)

        # 3. Winsorize Ratios
        ratio_cols = [c for c in df.columns if "rnd_to_rev_ratio" in c]
        if ratio_cols:
            df[ratio_cols] = df[ratio_cols].apply(self._winsorize_series)

        # 4. Log Assets
        if self.USE_LOG_ASSETS:
            for c in ["totalAssets", "totalEquity", "totalAssets_lag1", "totalEquity_lag1"]:
                if c in df.columns:
                    df[f"{c}_log"] = self._safe_log1p(df[c])

        # 5. Time Index (Crucial for TFT)
        df["time_idx"] = range(len(df))

        # 6. Drop early rows lacking minimum encoder features
        if self.DROP_INCOMPLETE_ENCODER_ROWS:
            df = df.dropna(subset=self.MINIMAL_ENCODER_COLS).reset_index(drop=True)

        return df

    # ==========================================
    # 3. EXECUTION PIPELINE
    # ==========================================
    def run(self):
        master_panel_frames = []
        missing = []

        for ticker in self.tickers:
            file_path = os.path.join(self.input_dir, f"{ticker}.csv")
            if not os.path.exists(file_path):
                missing.append(ticker)
                continue

            try:
                # Execute Pipeline
                df = pd.read_csv(file_path)
                df['ticker'] = ticker
                
                df = self.process_raw_data(df, ticker)
                df = self.engineer_historical_features(df)
                df = self.append_future_quarters(df, n_future=4)
                df = self.finalize_tft_features(df)

                # Save individual file
                out_path = os.path.join(self.output_dir, f"{ticker}_features.csv")
                df.to_csv(out_path, index=False)
                
                master_panel_frames.append(df)
                
            except Exception as e:
                print(f"[Error] {ticker}: {e}")

        # Compile Master Panel
        if master_panel_frames:
            master_df = pd.concat(master_panel_frames, ignore_index=True)
            master_df.to_csv(os.path.join(self.output_dir, "tft_master_panel.csv"), index=False)
            print(f"\nPipeline Complete. Master Panel saved with {len(master_df)} total rows.")
            
        if missing:
            print(f"Missing {len(missing)} source files. Saved log.")
            pd.Series(missing, name="missing_tickers").to_csv(os.path.join(self.output_dir, "_missing.csv"), index=False)


if __name__ == "__main__":
    # Ensure these paths map to your local environment
    INPUT_DIR = r"C:\ThesisResearch\finbert_tft\data\financial_data\fin_raw"
    OUTPUT_DIR = r"C:\ThesisResearch\finbert_tft\data\tft_ready_features"
    # Make sure to point this to a CSV containing your 7 tickers (e.g. column 'ticker' with AAPL, MSFT, META, etc.)
    TICKERS_CSV = r"C:\ThesisResearch\finbert_tft\data\tickers.csv"
    
    pipeline = IntegratedTFTFeaturePipeline(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        tickers_csv_path=TICKERS_CSV
    )
    pipeline.run()