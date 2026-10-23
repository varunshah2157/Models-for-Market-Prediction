import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = "SPY"
start_date = "2021-01-01"
end_date = "2024-01-01"
trading_days_per_year = 252
risk_free_rate = 0.0

df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']].copy()
df.dropna(inplace=True)

print(f"data loaded for {ticker}: {len(df)} trading days.")

df["Log_Return"] = np.log(df["Close"]/df["Close"].shift(1))
df.dropna(inplace=True)

plt.figure(figsize=(10, 6))
plt.hist(df['Log_Return'], bins=100, alpha=0.7,
         color='blue', edgecolor='black')
plt.title(f"Daily Log Return Distribution for {ticker}")
plt.xlabel("Daily Log Return")
plt.ylabel("Frequency")
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(alpha=0.3)
plt.show()

mean_daily_return = df["Log_Return"].mean()
annual_return = mean_daily_return*trading_days_per_year

daily_volatility = df["Log_Return"].std()
annual_volatility = daily_volatility*np.sqrt(trading_days_per_year)

annual_sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

df["Equity_Curve"] = np.exp(df["Log_Return"].cumsum())
df["Rolling_Max"] = df["Equity_Curve"].cummax()
df["Drawdown"] = (df["Equity_Curve"]-df["Rolling_Max"])/df["Rolling_Max"]
max_drawdown = df["Drawdown"].min()

best_day = df["Log_Return"].max()
worst_day = df["Log_Return"].min()

print(
    f"mean daily return: {mean_daily_return:.4f} ({mean_daily_return*100:.2f}%)")
print(f"annualized return: {annual_return:.4f} ({annual_return*100:.2f}%)")
print(
    f"annualized volatility: {annual_volatility:.4f} ({annual_volatility*100:.2f}%)")
print(f"annualized sharpe ratio: {annual_sharpe_ratio:.4f}")
print(f"maximum drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
print(f"best day: {best_day:.4f} ({best_day*100:.2f}%)")
print(f"worst day: {worst_day:.4f} ({worst_day*100:.2f}%)")
