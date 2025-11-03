# Quantitative Finance Toolkit

A Python-based implementation of quantitative finance models including options pricing (Black-Scholes, Monte Carlo), portfolio optimization (Modern Portfolio Theory), and statistical arbitrage (pairs trading). Built for educational purposes and quantitative analysis.

**Author:** Paridhi D Choudhary | IIT Kharagpur  
**Tech Stack:** Python, NumPy, Pandas, SciPy, Matplotlib, yfinance

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Options Pricing](#options-pricing)
3. [Portfolio Optimization](#portfolio-optimization)
4. [Statistical Arbitrage](#statistical-arbitrage)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Technical Notes](#technical-notes)
8. [Results](#results)

---

## 🎯 Overview

This project implements three core quantitative finance strategies:

### **1. Options Pricing**
- **Black-Scholes Model** for European options
- **Monte Carlo Simulation** for path-dependent options (European & Asian)
- **Greeks Calculation** (Delta, Gamma, Vega, Theta, Rho)

### **2. Portfolio Optimization**
- **Modern Portfolio Theory** (Markowitz mean-variance optimization)
- **Efficient Frontier** generation
- **Risk Metrics** (VaR, CVaR, Sharpe Ratio)

### **3. Statistical Arbitrage**
- **Pairs Trading** strategy
- **Cointegration Testing** (Engle-Granger method)
- **Mean Reversion** exploitation via z-score signals

---

## 📈 Options Pricing

### What It Does

Prices options using both analytical (Black-Scholes) and numerical (Monte Carlo) methods, then calculates sensitivity metrics (Greeks).

### Theory

#### **Black-Scholes Formula**

For a European call option:
```
C = S₀·N(d₁) - K·e^(-rT)·N(d₂)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
N(x) = Cumulative normal distribution
```

**Key Parameters:**
- `S₀` = Current stock price
- `K` = Strike price
- `T` = Time to expiration (years)
- `r` = Risk-free interest rate
- `σ` = Volatility (annualized)

#### **Why Normal CDF (N(d))?**

The CDF appears because stock prices follow **Geometric Brownian Motion**:
```
dS = μS dt + σS dW
```

This makes log returns normally distributed:
```
ln(S_T/S₀) ~ N((r - σ²/2)T, σ²T)
```

**N(d₂)** = Probability option expires in-the-money (ITM)  
**N(d₁)** = Delta-adjusted probability (expected value component)

#### **The Greeks (Sensitivities)**

| Greek | Formula | Meaning |
|-------|---------|---------|
| **Delta (Δ)** | `∂C/∂S = N(d₁)` | Price change per $1 stock move |
| **Gamma (Γ)** | `∂²C/∂S² = N'(d₁)/(S·σ·√T)` | Rate of Delta change |
| **Vega (ν)** | `∂C/∂σ = S·√T·N'(d₁)` | Price change per 1% volatility change |
| **Theta (Θ)** | `∂C/∂t` | Price decay per day (time decay) |
| **Rho (ρ)** | `∂C/∂r = K·T·e^(-rT)·N(d₂)` | Price change per 1% rate change |

**Greek Interpretations:**
```
Delta = 0.60 → $1 stock increase → $0.60 option increase
Gamma = 0.05 → Stock up $1 → Delta increases to 0.65
Theta = -$0.08 → Each day → Lose $0.08 to time decay
Vega = $0.38 → Volatility up 1% → Option price up $0.38
```

**Why Greeks Matter:**
- **Delta:** Position sizing and hedging
- **Gamma:** Risk of Delta changing (acceleration)
- **Theta:** Cost of holding options (time decay)
- **Vega:** Exposure to volatility changes
- **Rho:** Interest rate sensitivity (usually minimal)

#### **Monte Carlo Simulation**

**European Options:**
1. Simulate final stock prices: `S_T = S₀·e^((r-σ²/2)T + σ√T·Z)`
2. Calculate payoffs: `max(S_T - K, 0)` for calls
3. Average and discount: `C = e^(-rT)·E[payoff]`

**Asian Options:**
- Payoff depends on **average price** over option life
- More complex: must simulate entire price path
- **Cheaper than European** (averaging reduces volatility by ~1/√2)
```python
# Asian vs European pricing example
European Call: $10.45
Asian Call:    $5.75  (45% cheaper!)

Why? Volatility of average < volatility of final price
```

### Implementation Details
```python
class OptionsAnalyzer:
    def __init__(self, S0, K, T, r, sigma):
        # Store option parameters
        
    def call_price(self):
        # Black-Scholes analytical solution
        
    def greeks(self):
        # Calculate all sensitivities
        
    def monte_carlo_european(self, n_sims=100000):
        # Simulate final prices only
        
    def monte_carlo_asian(self, n_sims=50000, n_steps=252):
        # Simulate full price paths (252 trading days)
```

### Example Results
```
Parameters: S=$100, K=$100, T=1yr, r=5%, σ=20%

BLACK-SCHOLES:
  Call Price: $10.45
  Put Price:  $5.57

GREEKS:
  Call Delta: 0.64  (60%+ probability of expiring ITM)
  Gamma:      0.019 (Delta changes slowly, long-dated option)
  Vega:       $0.38 (significant vol sensitivity)
  Theta:      -$0.02/day (time decay)

MONTE CARLO (100k simulations):
  Call: $10.47 ± $0.05  (0.2% error vs Black-Scholes ✓)
  
ASIAN OPTIONS (50k simulations):
  Asian Call: $5.75 (45% cheaper due to volatility reduction)
```

**Key Insight:** Monte Carlo validates Black-Scholes for European options, but is essential for path-dependent options like Asians.

---

## 📊 Portfolio Optimization

### What It Does

Finds optimal asset allocations to maximize risk-adjusted returns using Modern Portfolio Theory (Markowitz, 1952).

### Theory

#### **Core Concepts**

**Portfolio Return (Simple):**
```
R_p = w₁R₁ + w₂R₂ + ... + wₙRₙ
    = Σ(w_i × R_i)
```
Weighted average of individual returns.

**Portfolio Risk (Complex):**
```
σ_p = √(w'Σw)

where Σ = covariance matrix
```
**NOT a simple average!** Incorporates correlations.

**Example (2 assets):**
```
Asset A: 20% return, 30% volatility
Asset B: 10% return, 15% volatility
Correlation: 0.5

50-50 portfolio:
  Return: 15% (simple average ✓)
  Risk: 19.4% (NOT 22.5% average!)
  
Risk reduction: (22.5% - 19.4%) / 22.5% = 13.8%
```

#### **The Diversification Benefit**

**Correlation Impact:**
```
ρ = +1.0 (perfect positive):
  → Portfolio risk = weighted average (NO benefit)
  
ρ = 0.0 (uncorrelated):
  → Portfolio risk reduced by 1/√n
  → 2 assets: risk × 0.707 (29% reduction)
  
ρ = -1.0 (perfect negative):
  → Portfolio risk → 0 (perfect hedge!)
```

**Real Example:**
```
Tech stocks (AAPL, MSFT, GOOGL):
  Correlation: 0.75-0.85 (high)
  → Limited diversification benefit
  
Tech + Bonds:
  Correlation: 0.10-0.30 (low)
  → Strong diversification benefit
```

#### **Efficient Frontier**

The **Efficient Frontier** is the set of portfolios with:
- Maximum return for given risk, OR
- Minimum risk for given return
```
        Return
          ^
          |         * Efficient Frontier
      20% |      *
          |    *   Individual
      15% | *    * stocks
          |*   *
      10% | * * Random portfolios
          |*
       5% |
          +-----------------------> Risk
            5%   10%   15%   20%
```

**Key Points:**

1. **Maximum Sharpe Ratio Portfolio (Red Star)**
   - Best risk-adjusted returns
   - Typically 60-80% of way along frontier
   
2. **Minimum Volatility Portfolio (Green Star)**
   - Lowest possible risk
   - Sacrifices some return for safety

3. **Random Portfolios (Cloud)**
   - Inefficient (dominated by frontier points)

#### **Optimization Problem**

**Maximize Sharpe Ratio:**
```
max (R_p - r_f) / σ_p

subject to:
  Σw_i = 1      (weights sum to 100%)
  w_i ≥ 0       (no short selling)
  0 ≤ w_i ≤ 1   (bounded positions)
```

**Minimize Volatility:**
```
min √(w'Σw)

subject to same constraints
```

#### **Risk Metrics**

**Value at Risk (VaR):**
> "Maximum expected loss at X% confidence level"
```
95% VaR = -3.2%
→ 95% of days, losses won't exceed 3.2%
→ 1 in 20 days will be worse
```

**Conditional VaR (CVaR):**
> "Average loss when VaR threshold is exceeded"
```
95% CVaR = -5.4%
→ When losses exceed VaR, they average 5.4%
→ Measures tail risk
```

**Sharpe Ratio:**
```
Sharpe = (R_p - r_f) / σ_p

Interpretation:
  < 0.5: Poor
  0.5-1.0: Good
  1.0-2.0: Very good
  > 2.0: Exceptional (rare)
```

### Implementation Details
```python
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date):
        # Download historical data
        
    def portfolio_stats(self, weights):
        # Calculate return, volatility, Sharpe
        
    def optimize_max_sharpe(self):
        # Use SciPy minimize with constraints
        
    def optimize_min_vol(self):
        # Minimize volatility subject to constraints
        
    def efficient_frontier(self, n_portfolios=5000):
        # Generate random portfolios for visualization
```

### Example Results
```
MAXIMUM SHARPE RATIO PORTFOLIO:
  Allocation: 60% AAPL, 40% AMZN
  Return: 28.97%
  Volatility: 31.36%
  Sharpe: 0.92
  95% VaR: -47.07%
  95% CVaR: -69.45%

MINIMUM VOLATILITY PORTFOLIO:
  Allocation: 100% SPY
  Return: 13.69%
  Volatility: 22.63%
  Sharpe: 0.61
  95% VaR: -32.35%
  95% CVaR: -54.54%
```

**Analysis:**
- Max Sharpe concentrates in winners (AAPL, AMZN) → High return but extreme tail risk
- Min Vol chooses broad index (SPY) → Lower return but safer
- **Period dependency:** These results reflect 2020-2024 tech boom, not general truth

**Key Insight:** The optimizer is backward-looking. It tells you what WAS optimal, not what WILL BE optimal. Always consider regime changes!

---

## 🔄 Statistical Arbitrage (Pairs Trading)

### What It Does

Exploits temporary price divergences between cointegrated assets through market-neutral strategies.

### Theory

#### **Core Concepts**

**Cointegration:**
> "Two assets that wander apart temporarily but are pulled back together by economic forces"

**Example:**
```
Coca-Cola (KO) and Pepsi (PEP):
- Same industry (beverages)
- Same customers
- Same competitors
- Same supply chain
→ Prices move together long-term

Like two people on a leash:
  Person A: ←→←→  (KO)
  Person B: ←→←→  (PEP)
            ↕
          Leash (economic relationship)
```

**Cointegration vs Correlation:**
```
Correlation = "Do they move together RIGHT NOW?"
  - Can be high temporarily then disappear
  - Not tradeable (no reversion guarantee)
  
Cointegration = "Do they have a STABLE long-term relationship?"
  - Ratio mean-reverts
  - Tradeable! (divergences are temporary)
```

#### **Statistical Testing**

**Engle-Granger Cointegration Test:**

1. Calculate spread: `spread = Price_A / Price_B`
2. Test if spread is stationary (doesn't drift to infinity)
3. Get p-value

**P-Value Interpretation:**
```
P-value = Probability of seeing this relationship by random chance

p < 0.05 → Cointegrated ✓ (< 5% chance it's random)
p ≥ 0.05 → NOT cointegrated ✗ (might be random)
```

**Example:**
```
KO vs PEP:
  Cointegration Test Statistic: -3.21
  P-value: 0.0156 (1.56%)
  
Interpretation: 
  Only 1.56% chance this relationship is random
  → 98.44% confident they're cointegrated ✓
```

#### **The Trading Strategy**

**1. Calculate Spread:**
```
Spread = Stock_A / Stock_B
```

**2. Calculate Z-Score:**
```
Z = (Spread - Mean_Spread) / Std_Dev_Spread
```

**Z-Score Interpretation:**
```
Z = +2.0 → Spread is 2 std dev ABOVE average
         → Stock A expensive, Stock B cheap
         → SHORT spread (short A, long B)

Z = -2.0 → Spread is 2 std dev BELOW average
         → Stock A cheap, Stock B expensive
         → LONG spread (long A, short B)

Z → 0   → Spread returned to normal
         → CLOSE position, take profit
```

**3. Entry/Exit Rules:**
```
Entry: |Z| > 2.0 (divergence)
Exit:  |Z| < 0.5 (convergence)
```

#### **Why It's Market Neutral**
```
Regular investing:
  Buy AAPL → Exposed to market risk
  Market crashes -20% → You lose 20%

Pairs trading:
  Long $50k AAPL, Short $50k MSFT
  Market crashes -20%:
    AAPL: -$10k
    MSFT short: +$10k
    Net: $0 ✓
    
Only RELATIVE performance matters!
```

#### **Example Trade**
```
Day 1: Analysis
-------
KO/PEP historical ratio: 0.55 ± 0.01
Current ratio: 0.574
Z-score: (0.574 - 0.55) / 0.01 = +2.4

Day 1: Enter Trade (Z > 2.0)
-------
SHORT $50k KO at $58  (862 shares)
LONG  $50k PEP at $101 (495 shares)

Day 10: Monitor
-------
KO: $56 (down 3.4%)
PEP: $103 (up 2.0%)
Ratio: 0.544
Z-score: -0.6

Day 10: Exit Trade (|Z| < 0.5 crossed)
-------
Cover KO: Buy at $56 → Profit: ($58-$56) × 862 = $1,724
Sell PEP: Sell at $103 → Profit: ($103-$101) × 495 = $990

Total profit: $2,714 on $100k capital = 2.71%
Time: 9 days
Annualized: ~110% (if this continued)
```

### Implementation Details
```python
class PairsTradingStrategy:
    def test_cointegration(self):
        # Engle-Granger test
        score, pval, _ = coint(stock1, stock2)
        return pval < 0.05
        
    def calculate_spread(self):
        spread = stock1 / stock2
        z_score = (spread - spread.mean()) / spread.std()
        
    def backtest(self, entry_z=2.0, exit_z=0.5):
        # Entry: |z| > entry_z
        # Exit: |z| < exit_z
        # Calculate returns: position × (R1 - R2)
```

### Example Results
```
Pair: KO vs PEP (2020-2024)

Cointegration Test:
  P-value: 0.0156 ✓ Cointegrated
  
Performance:
  Total Return: 8.45%
  Annual Return: 2.07%
  Sharpe Ratio: 1.23  (Excellent!)
  Max Drawdown: -3.21% (Very low!)
  Win Rate: 62.5%
  Total Trades: 24
```

**Analysis:**
- **Low returns (2% annually)** but **very low risk** (3% max drawdown)
- **High Sharpe (1.23)** → Excellent risk-adjusted performance
- **Market neutral** → Protected from crashes
- **Consistent** → 62.5% win rate

**Key Insight:** Pairs trading sacrifices absolute returns for consistency and low risk. It's not about getting rich quick - it's about steady, reliable profits with minimal drawdown.

---


## 🚀 Usage

### Options Pricing
```python
from options_pricing import OptionsAnalyzer

# Create option
opt = OptionsAnalyzer(S0=100, K=100, T=1.0, r=0.05, sigma=0.20)

# Black-Scholes pricing
call_price = opt.call_price()
put_price = opt.put_price()

# Greeks
greeks = opt.greeks()
print(f"Delta: {greeks['call_delta']:.4f}")

# Monte Carlo
mc_results = opt.monte_carlo_european(n_sims=100000)
print(f"Call: ${mc_results['call']:.2f}")

# Visualize
opt.plot_analysis()
```

### Portfolio Optimization
```python
from portfolio_optimization import PortfolioOptimizer

# Create optimizer
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
optimizer = PortfolioOptimizer(tickers, start='2020-01-01', end='2024-01-01')

# Download data
optimizer.fetch_data()

# Find optimal portfolios
max_sharpe_weights, min_vol_weights = optimizer.plot_efficient_frontier()

# Analyze
optimizer.print_summary(max_sharpe_weights, "Max Sharpe Portfolio")
```

### Statistical Arbitrage
```python
from statistical_arbitrage import PairsTradingStrategy

# Create strategy
pairs = PairsTradingStrategy('KO', 'PEP', start='2020-01-01', end='2024-01-01')

# Test cointegration
pairs.fetch_data()
is_coint, pval = pairs.test_cointegration()

if is_coint:
    # Backtest
    pairs.calculate_spread()
    signals = pairs.backtest(entry_z=2.0, exit_z=0.5)
    
    # Evaluate
    metrics = pairs.calculate_metrics(signals)
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    
    # Visualize
    pairs.plot_analysis(signals)
```

---

## 📝 Technical Notes

### Mathematical Foundations

#### **Stochastic Calculus**
```
Stock price follows Geometric Brownian Motion:
dS = μS dt + σS dW

Solution (Ito's Lemma):
S_t = S_0 × exp((μ - σ²/2)t + σW_t)

This is why log returns are normally distributed:
ln(S_t/S_0) ~ N((μ - σ²/2)t, σ²t)
```

#### **Risk-Neutral Pricing**
```
Under risk-neutral measure:
μ → r (drift becomes risk-free rate)

Option price = e^(-rT) × E^Q[Payoff]

This is the foundation of Black-Scholes!
```

#### **Quadratic Programming**
```
Portfolio optimization solves:
min w'Σw
subject to: w'μ ≥ R_target, Σw_i = 1, w_i ≥ 0

This is a convex optimization problem
→ Unique global optimum exists
```

### Numerical Methods

**Monte Carlo Convergence:**
```
Standard Error ∝ 1/√n

n = 10,000:   SE ≈ 0.06
n = 100,000:  SE ≈ 0.02
n = 1,000,000: SE ≈ 0.006

10x more sims → 3.16x less error (√10 ≈ 3.16)
```

**Optimization Algorithms:**
- **SLSQP** (Sequential Least Squares Programming)
- Handles constraints efficiently
- Converges to local minimum (but problem is convex, so local = global)

### Common Pitfalls

**1. Look-Ahead Bias:**
```python
# WRONG:
signals['pos'].iloc[i] = f(data.iloc[i])  # Using same-day data

# CORRECT:
signals['pos'].iloc[i] = f(data.iloc[i-1])  # Using previous day
signals['returns'].iloc[i] = signals['pos'].iloc[i-1] * returns.iloc[i]
```

**2. Overfitting:**
```
Testing on same data used for optimization
→ Unrealistic performance estimates

Solution: Use train/test split or walk-forward analysis
```

**3. Transaction Costs:**
```
Ignoring commissions and slippage
→ Overestimated returns

Reality: Subtract ~0.1-0.5% per trade
```

**4. Survivorship Bias:**
```
Only testing on stocks that still exist today
→ Misses companies that went bankrupt

Solution: Use "survivorship-bias-free" datasets
```

---

## 📊 Results Summary

### Options Pricing
- Black-Scholes vs Monte Carlo: < 0.3% error (validated ✓)
- Asian options 45% cheaper than European (volatility reduction)
- Greeks match theoretical expectations

### Portfolio Optimization
- Max Sharpe (2020-2024): 29% annual return, 0.92 Sharpe
- Concentration risk: 60% AAPL + 40% AMZN (period-dependent!)
- Min Vol: 14% return, 0.61 Sharpe, much lower drawdown

### Statistical Arbitrage
- KO-PEP pair: 8.45% total return, 1.23 Sharpe
- Max drawdown only 3.21% (excellent risk control)
- 62.5% win rate, 24 trades over 4 years

---

## 🎓 Learning Resources

### Books
- **Options:** "Options, Futures, and Other Derivatives" - John Hull
- **Portfolio Theory:** "Portfolio Selection" - Harry Markowitz
- **Quant Trading:** "Quantitative Trading" - Ernest Chan
- **Interview Prep:** "A Practical Guide To Quantitative Finance Interviews" (Green Book)

### Papers
- Black-Scholes (1973): "The Pricing of Options and Corporate Liabilities"
- Markowitz (1952): "Portfolio Selection"
- Engle-Granger (1987): "Co-integration and Error Correction"

---

## 🤝 Contributing

This is an educational project. Suggestions for improvements:
- Add more exotic options (barriers, lookbacks)
- Implement regime-switching models
- Add transaction costs to backtests
- Walk-forward optimization
- Risk parity allocation

---


## 📧 Contact

**Paridhi D Choudhary**  
IIT Kharagpur | B.Tech (Hons.) Aerospace Engineering | M.Tech AI/ML  
CGPA: 9.22/10

For questions or collaboration: [paridhidchoudhary@gmail.com/https://www.linkedin.com/in/paridhi-d-choudhary-161632232/]

---


