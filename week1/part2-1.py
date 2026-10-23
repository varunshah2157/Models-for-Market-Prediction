import yfinance as yf
import pandas as pd
import numpy as np

ticker = "SPY"
start_date = "2021-01-01"
end_date = "2024-01-01"
trading_days_per_year = 252

df = yf.download(ticker, start=start_date, end=end_date)
df = df['Close'].copy()

if isinstance(df, pd.Series):
    df = df.to_frame(name='Close')
else:
    df.columns = ['Close']

df.dropna(inplace=True)
print(f"data loaded for {ticker}: {len(df)} trading days.")

df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

df['SMA_60'] = df['Close'].rolling(window=60).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()

condition_uptrend = df['SMA_20'] > df['SMA_60']
condition_dip = df['Close'] < df['SMA_10']

df['Signal'] = np.where(condition_uptrend & condition_dip, 1, 0)
df['Position'] = df['Signal'].shift(1)

df_strat = df.dropna().copy()

df_strat['Strat_Log_Return'] = df_strat['Position'] * df_strat['Log_Return']

df_strat['Fwd_5D_Return'] = df_strat['Log_Return'].rolling(
    window=5).sum().shift(-5)

avg_5d_return_on = df_strat[df_strat['Signal'] == 1]['Fwd_5D_Return'].mean()
avg_5d_return_off = df_strat[df_strat['Signal'] == 0]['Fwd_5D_Return'].mean()

print("\n--- 5-DAY FORWARD RETURNS ---")
print(
    f"Avg 5-Day Return when Signal ON:  {avg_5d_return_on:.4f} ({avg_5d_return_on*100:.2f}%)")
print(
    f"Avg 5-Day Return when Signal OFF: {avg_5d_return_off:.4f} ({avg_5d_return_off*100:.2f}%)")
print("Direction matches hypothesis? ",
      "Yes" if avg_5d_return_on > avg_5d_return_off else "No")
print("\n")

strat_mean_daily = df_strat['Strat_Log_Return'].mean()
strat_ann_return = strat_mean_daily * trading_days_per_year
strat_ann_volatility = df_strat['Strat_Log_Return'].std(
) * np.sqrt(trading_days_per_year)
strat_ann_sharpe_ratio = strat_ann_return / \
    strat_ann_volatility if strat_ann_volatility != 0 else 0

df_strat['Strat_Equity'] = np.exp(df_strat['Strat_Log_Return'].cumsum())
df_strat['Strat_Roll_Max'] = df_strat['Strat_Equity'].cummax()
df_strat['Strat_Drawdown'] = (
    df_strat['Strat_Equity'] - df_strat['Strat_Roll_Max']) / df_strat['Strat_Roll_Max']
strat_max_drawdown = df_strat['Strat_Drawdown'].min()

bench_mean_daily = df_strat['Log_Return'].mean()
bench_ann_return = bench_mean_daily * trading_days_per_year
bench_ann_volatility = df_strat['Log_Return'].std(
) * np.sqrt(trading_days_per_year)
bench_ann_sharpe_ratio = bench_ann_return / bench_ann_volatility

df_strat['Bench_Equity'] = np.exp(df_strat['Log_Return'].cumsum())
df_strat['Bench_Roll_Max'] = df_strat['Bench_Equity'].cummax()
df_strat['Bench_Drawdown'] = (
    df_strat['Bench_Equity'] - df_strat['Bench_Roll_Max']) / df_strat['Bench_Roll_Max']
bench_max_drawdown = df_strat['Bench_Drawdown'].min()

print("--- STRATEGY VS BENCHMARK ---")
comparison_data = {
    'Metric': ['Annualized Return', 'Annualized Volatility', 'Annualized Sharpe', 'Max Drawdown'],
    'Benchmark (Buy & Hold)': [f"{bench_ann_return*100:.2f}%", f"{bench_ann_volatility*100:.2f}%", f"{bench_ann_sharpe_ratio:.4f}", f"{bench_max_drawdown*100:.2f}%"],
    'Strategy (Regime Filter)': [f"{strat_ann_return*100:.2f}%", f"{strat_ann_volatility*100:.2f}%", f"{strat_ann_sharpe_ratio:.4f}", f"{strat_max_drawdown*100:.2f}%"]
}

comp_df = pd.DataFrame(comparison_data)
print(comp_df.to_string(index=False))
