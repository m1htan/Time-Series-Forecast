import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os

class StockAnalyzer:
    def __init__(self, tickers, start_date, end_date, output_dir='output'):
        """Initialize the StockAnalyzer with tickers and date range."""
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.data = None
        self.adj_close = None
        self.daily_returns = None
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory

    def download_data(self):
        """Download stock data from Yahoo Finance."""
        try:
            # Set auto_adjust=False to include Adjusted Close column
            self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date,
                                   group_by='ticker', auto_adjust=False)
            if self.data.empty:
                raise ValueError("No data was downloaded. Check internet connection or ticker symbols.")
            print("Available columns in downloaded data:")
            for ticker in self.tickers:
                print(f"{ticker}: {self.data[ticker].columns.tolist()}")
        except Exception as e:
            raise Exception(f"Error downloading data: {e}")

    def preprocess_data(self):
        """Preprocess data to extract adjusted close prices and compute daily returns."""
        # Determine the correct column for adjusted close prices
        possible_columns = ['Adjusted Close', 'Adj Close', 'Close']
        adj_close_column = None
        for col in possible_columns:
            if col in self.data[self.tickers[0]].columns:
                adj_close_column = col
                break
        if adj_close_column is None:
            raise KeyError(f"No valid price column found. Available columns: {self.data[self.tickers[0]].columns.tolist()}")

        print(f"Using column '{adj_close_column}' for adjusted close prices.")

        try:
            self.adj_close = pd.DataFrame({
                ticker: self.data[ticker][adj_close_column] for ticker in self.tickers
            })
        except KeyError as e:
            raise KeyError(f"Error accessing column '{adj_close_column}'. Available columns: {self.data[self.tickers[0]].columns.tolist()}")

        # Handle missing values
        self.adj_close.fillna(method='ffill', inplace=True)
        self.adj_close.fillna(method='bfill', inplace=True)
        print("\nMissing values in Adjusted Close data:")
        print(self.adj_close.isna().sum())

        # Calculate daily returns
        self.daily_returns = self.adj_close.pct_change().dropna()
        print("\nBasic Statistics of Daily Returns:")
        print(self.daily_returns.describe())

    def plot_adjusted_close(self):
        """Plot adjusted close prices for all tickers."""
        plt.figure(figsize=(14, 7))
        for ticker in self.tickers:
            plt.plot(self.adj_close.index, self.adj_close[ticker], label=ticker)
        plt.title('Adjusted Close Prices (2020-2025)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/adjusted_close_prices.png')
        plt.close()

    def plot_returns_distribution(self):
        """Plot distribution of daily returns for all tickers."""
        plt.figure(figsize=(14, 7))
        for ticker in self.tickers:
            sns.histplot(self.daily_returns[ticker], label=ticker, kde=True, bins=50)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'{self.output_dir}/daily_returns_distribution.png')
        plt.close()

    def plot_correlation_matrix(self):
        """Plot correlation matrix of daily returns."""
        correlation_matrix = self.daily_returns.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Daily Returns')
        plt.savefig(f'{self.output_dir}/correlation_matrix.png')
        plt.close()
        return correlation_matrix

    def plot_rolling_volatility(self, window=30):
        """Plot 30-day rolling volatility (annualized)."""
        rolling_volatility = self.daily_returns.rolling(window=window).std() * np.sqrt(252)
        plt.figure(figsize=(14, 7))
        for ticker in self.tickers:
            plt.plot(rolling_volatility.index, rolling_volatility[ticker], label=ticker)
        plt.title(f'{window}-Day Rolling Volatility (Annualized)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/rolling_volatility.png')
        plt.close()
        return rolling_volatility

    def plot_moving_averages(self, short_window=20, long_window=50):
        """Plot stock prices with short and long moving averages."""
        plt.figure(figsize=(14, 7))
        for ticker in self.tickers:
            ma_short = self.adj_close[ticker].rolling(window=short_window).mean()
            ma_long = self.adj_close[ticker].rolling(window=long_window).mean()
            plt.plot(self.adj_close.index, self.adj_close[ticker], label=f'{ticker} Price')
            plt.plot(ma_short.index, ma_short, label=f'{ticker} {short_window}-Day MA')
            plt.plot(ma_long.index, ma_long, label=f'{ticker} {long_window}-Day MA')
        plt.title(f'Stock Prices with {short_window}-Day and {long_window}-Day Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/moving_averages.png')
        plt.close()

    def plot_timeseries_decomposition(self):
        """Perform and plot time-series decomposition for each ticker."""
        for ticker in self.tickers:
            decomposition = seasonal_decompose(self.adj_close[ticker], model='multiplicative', period=252)
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f'Time-Series Decomposition of {ticker} Adjusted Close')
            ax1 = fig.add_subplot(411)
            ax1.plot(decomposition.observed, label='Observed')
            ax1.legend()
            ax1.grid(True)
            ax2 = fig.add_subplot(412)
            ax2.plot(decomposition.trend, label='Trend')
            ax2.legend()
            ax2.grid(True)
            ax3 = fig.add_subplot(413)
            ax3.plot(decomposition.seasonal, label='Seasonal')
            ax3.legend()
            ax3.grid(True)
            ax4 = fig.add_subplot(414)
            ax4.plot(decomposition.resid, label='Residual')
            ax4.legend()
            ax4.grid(True)
            plt.savefig(f'{self.output_dir}/timeseries_decomposition_{ticker}.png')
            plt.close()

    def calculate_rsi(self, periods=14):
        """Calculate Relative Strength Index (RSI) for each ticker."""
        rsi_data = pd.DataFrame()
        for ticker in self.tickers:
            delta = self.adj_close[ticker].diff()
            gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_data[ticker] = rsi
        return rsi_data

    def plot_rsi(self, periods=14):
        """Plot RSI for each ticker."""
        rsi_data = self.calculate_rsi(periods)
        plt.figure(figsize=(14, 7))
        for ticker in self.tickers:
            plt.plot(rsi_data.index, rsi_data[ticker], label=ticker)
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title(f'{periods}-Day Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/rsi.png')
        plt.close()

    def calculate_bollinger_bands(self, window=20, num_std=2):
        """Calculate Bollinger Bands for each ticker."""
        bollinger_data = {}
        for ticker in self.tickers:
            rolling_mean = self.adj_close[ticker].rolling(window=window).mean()
            rolling_std = self.adj_close[ticker].rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            bollinger_data[ticker] = pd.DataFrame({
                'Price': self.adj_close[ticker],
                'Middle Band': rolling_mean,
                'Upper Band': upper_band,
                'Lower Band': lower_band
            })
        return bollinger_data

    def plot_bollinger_bands(self, window=20, num_std=2):
        """Plot Bollinger Bands for each ticker."""
        bollinger_data = self.calculate_bollinger_bands(window, num_std)
        for ticker in self.tickers:
            plt.figure(figsize=(14, 7))
            plt.plot(bollinger_data[ticker].index, bollinger_data[ticker]['Price'], label=f'{ticker} Price')
            plt.plot(bollinger_data[ticker].index, bollinger_data[ticker]['Middle Band'], label='Middle Band (SMA)')
            plt.plot(bollinger_data[ticker].index, bollinger_data[ticker]['Upper Band'], label='Upper Band')
            plt.plot(bollinger_data[ticker].index, bollinger_data[ticker]['Lower Band'], label='Lower Band')
            plt.fill_between(bollinger_data[ticker].index, bollinger_data[ticker]['Lower Band'],
                             bollinger_data[ticker]['Upper Band'], alpha=0.1)
            plt.title(f'Bollinger Bands for {ticker} (Window={window}, Std={num_std})')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/bollinger_bands_{ticker}.png')
            plt.close()

    def plot_cumulative_returns(self):
        """Plot cumulative returns for all tickers."""
        cumulative_returns = (1 + self.daily_returns).cumprod() - 1
        plt.figure(figsize=(14, 7))
        for ticker in self.tickers:
            plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)
        plt.title('Cumulative Returns (2020-2025)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/cumulative_returns.png')
        plt.close()
        return cumulative_returns

    def plot_monthly_decomposition(self):
        """Perform and plot time-series decomposition based on monthly average adjusted close prices."""
        for ticker in self.tickers:
            monthly_data = self.adj_close[ticker].resample('M').mean()

            # Drop missing values caused by resampling
            monthly_data.dropna(inplace=True)

            try:
                decomposition = seasonal_decompose(monthly_data, model='multiplicative', period=12)  # monthly seasonality
                fig = plt.figure(figsize=(14, 10))
                fig.suptitle(f'Monthly Time-Series Decomposition of {ticker}')
                ax1 = fig.add_subplot(411)
                ax1.plot(decomposition.observed, label='Observed')
                ax1.legend()
                ax1.grid(True)
                ax2 = fig.add_subplot(412)
                ax2.plot(decomposition.trend, label='Trend')
                ax2.legend()
                ax2.grid(True)
                ax3 = fig.add_subplot(413)
                ax3.plot(decomposition.seasonal, label='Seasonal')
                ax3.legend()
                ax3.grid(True)
                ax4 = fig.add_subplot(414)
                ax4.plot(decomposition.resid, label='Residual')
                ax4.legend()
                ax4.grid(True)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(f'{self.output_dir}/monthly_decomposition_{ticker}.png')
                plt.close()
            except Exception as e:
                print(f"Decomposition failed for {ticker}: {e}")

    def generate_insights(self):
        """Generate detailed insights for all tickers."""
        insights = []

        # 1. Price Trends and Total Returns
        insights.append("1. **Price Trends and Total Returns**:")
        for ticker in self.tickers:
            start_price = self.adj_close[ticker].iloc[0]
            end_price = self.adj_close[ticker].iloc[-1]
            total_return = (end_price - start_price) / start_price * 100
            insights.append(f"   - {ticker}: Total return from {self.start_date} to {self.end_date} is {total_return:.2f}%")

        # 2. Volatility
        insights.append("\n2. **Volatility**:")
        rolling_volatility = self.daily_returns.rolling(window=30).std() * np.sqrt(252)
        for ticker in self.tickers:
            avg_volatility = rolling_volatility[ticker].mean()
            max_volatility = rolling_volatility[ticker].max()
            insights.append(f"   - {ticker}: Average annualized volatility is {avg_volatility:.2%}, "
                            f"with a peak of {max_volatility:.2%}")

        # 3. Correlation
        insights.append("\n3. **Correlation**:")
        correlation_matrix = self.daily_returns.corr()
        for i, ticker1 in enumerate(self.tickers):
            for ticker2 in self.tickers[i+1:]:
                corr = correlation_matrix.loc[ticker1, ticker2]
                insights.append(f"   - Correlation between {ticker1} and {ticker2}: {corr:.2f}")

        # 4. Seasonality and Trends
        insights.append("\n4. **Seasonality and Trends**:")
        for ticker in self.tickers:
            decomposition = seasonal_decompose(self.adj_close[ticker], model='multiplicative', period=252)
            trend_strength = (decomposition.trend.max() - decomposition.trend.min()) / decomposition.trend.mean()
            insights.append(f"   - {ticker}: Shows a trend strength of {trend_strength:.2%}, with "
                            f"seasonal patterns potentially tied to product cycles or market events.")

        # 5. Moving Averages
        insights.append("\n5. **Moving Averages**:")
        for ticker in self.tickers:
            ma20 = self.adj_close[ticker].rolling(window=20).mean()
            ma50 = self.adj_close[ticker].rolling(window=50).mean()
            latest_ma20 = ma20.iloc[-1]
            latest_ma50 = ma50.iloc[-1]
            insights.append(f"   - {ticker}: Latest 20-day MA ({latest_ma20:.2f}) vs 50-day MA ({latest_ma50:.2f}). "
                            f"{'Bullish' if latest_ma20 > latest_ma50 else 'Bearish'} trend indicated.")

        # 6. RSI Analysis
        insights.append("\n6. **Relative Strength Index (RSI)**:")
        rsi_data = self.calculate_rsi()
        for ticker in self.tickers:
            latest_rsi = rsi_data[ticker].iloc[-1]
            status = ("Overbought" if latest_rsi > 70 else
                      "Oversold" if latest_rsi < 30 else "Neutral")
            insights.append(f"   - {ticker}: Latest RSI is {latest_rsi:.2f} ({status}).")

        # 7. Bollinger Bands
        insights.append("\n7. **Bollinger Bands**:")
        bollinger_data = self.calculate_bollinger_bands()
        for ticker in self.tickers:
            latest_price = bollinger_data[ticker]['Price'].iloc[-1]
            upper_band = bollinger_data[ticker]['Upper Band'].iloc[-1]
            lower_band = bollinger_data[ticker]['Lower Band'].iloc[-1]
            position = ("Above Upper Band (Potential Overbought)" if latest_price > upper_band else
                        "Below Lower Band (Potential Oversold)" if latest_price < lower_band else
                        "Within Bands (Stable)")
            insights.append(f"   - {ticker}: Latest price ({latest_price:.2f}) is {position}.")

        return insights

    def save_data(self):
        """Save adjusted close prices and daily returns to CSV."""
        self.adj_close.to_csv(f'{self.output_dir}/stock_prices.csv')
        self.daily_returns.to_csv(f'{self.output_dir}/daily_returns.csv')
        print(f"\nData saved to '{self.output_dir}/stock_prices.csv' and '{self.output_dir}/daily_returns.csv'")

    def run_analysis(self):
        """Run the full analysis pipeline."""
        self.download_data()
        self.preprocess_data()
        print("\nBasic Statistics of Adjusted Close Prices:")
        print(self.adj_close.describe())
        self.plot_adjusted_close()
        self.plot_returns_distribution()
        self.plot_monthly_decomposition()
        self.plot_correlation_matrix()
        self.plot_rolling_volatility()
        self.plot_moving_averages()
        self.plot_timeseries_decomposition()
        self.plot_rsi()
        self.plot_bollinger_bands()
        self.plot_cumulative_returns()
        insights = self.generate_insights()
        print("\nInsights and Observations:")
        for insight in insights:
            print(insight)
        self.save_data()

if __name__ == "__main__":
    # Initialize and run analysis
    warnings.filterwarnings('ignore')
    tickers = ['AAPL', 'MSFT', 'GOOG']
    start_date = '2020-01-01'
    end_date = '2025-05-16'
    analyzer = StockAnalyzer(tickers, start_date, end_date)
    analyzer.run_analysis()