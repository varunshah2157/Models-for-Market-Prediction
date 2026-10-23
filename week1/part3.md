### Part 3: Honest Assessment & Strategy Evolution

#### Attempt 1: Three Moving Averages
* **Logic:** Buy when the 20-day average is above the 60-day average, AND the daily price dips below the 10-day average.
* **Output:** Annualized Return: -0.45% | Max Drawdown: -20.76% | Sharpe: -0.0495
* **Why it failed:** Moving averages are slow to update. By the time my rules finally triggered a "buy" signal, the market was often already in the middle of a crash. I was buying at the worst possible times.

---

#### Attempt 2: Simple Average Crossover
* **Logic:** Removed the 10-day rule. Just buy when the 20-day average crosses above the 60-day average.
* **Output:** Annualized Return: -0.06% | Max Drawdown: -26.38% | Sharpe: -0.0049
* **Why it failed:** The signal was always late. When the price went up and down frequently, the delay caused me to buy high and sell low. The maximum loss (-26.38%) was actually worse than just buying and holding SPY (-24.50%).

---

#### Attempt 3: The 200-Day Trend Filter (Final)
* **Logic:** Buy when the daily price drops below the 20-day average, but only if the long-term trend is upward (price is above the 200-day average).
* **Output:** Annualized Return: -0.05% | Max Drawdown: -9.47% | Sharpe: -0.0066
* **Why it failed:** The rules were too strict. By trying so hard to avoid market crashes, the strategy kept my money out of the market (in cash) for most of the three years, completely missing the big upward rallies. Furthermore, when it actually did buy a "dip," the stock often just continued to drop slightly, resulting in small, repeated losses rather than catching a bounce.

---

#### Final Assessment

**1. Live Trading Viability**
The single biggest reason this result wouldn't work in live trading is the simplest one: it couldn't even make a profit in a perfect simulation. A backtest is the "best-case scenario" because it has no real-world friction. If I am not even able to make profits on 3 years of known, historical data, it is absolutely not viable to trade in live, unpredictable markets. Taking a strategy that already produces negative returns (-0.05%) on paper and adding real-world costs like broker fees, bid-ask spreads, and slippage will only cause it to lose money even faster. 

**2. Independent Observations & Statistical Trust**
My test covers 753 trading days, but my strategy relies on both 20-day and 200-day moving averages. Because today's average shares almost all of its data with yesterday's, the days overlap heavily and are not unique events. The reading states we must divide the total days by the window size to find truly independent observations:
* For my short-term 20-day rule, I only have about **37 independent events** (753 ÷ 20). 
* More severely, for my 200-day long-term rule, I only have about **3.7 independent sections of data** (753 ÷ 200). 

Calculating a Sharpe ratio over so few genuinely independent data points means the metric is mathematically unreliable and highly likely to just be random luck.

**3. Strategy Evaluation: Tradeoff vs. Improvement**
When comparing the metrics, this strategy is technically a tradeoff, but practically, it is strictly worse on both fronts when evaluated as an investment. 
* **Drawdown (Improvement):** The strategy successfully reduced risk, dropping the maximum drawdown from -24.50% (Buy and Hold Benchmark) to just -9.47%.
* **Sharpe & Returns (Worse):** This risk reduction destroyed the returns. The Annualized Return fell from 4.49% to -0.05%, which dragged the Annualized Sharpe Ratio down from 0.2355 to -0.0066. 

By forcing the strategy to sit in cash for long periods to avoid drawdowns, it missed the market rallies. It is strictly worse than the benchmark because keeping the capital in a zero-interest checking account would have provided a better Sharpe ratio and a 0% drawdown.