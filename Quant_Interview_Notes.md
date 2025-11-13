
# 📘 Quant Interview Notes – Portfolio Optimization, Options Pricing, and Pairs Trading

Prepared as an interview-day refresher. Covers all key intuitions, math, and connections.

---

## ⚖️ 1. Options Pricing

### 💭 Intuition
An **option** is a contract that gives the **right (not obligation)** to buy or sell an asset at a **strike price (K)** before **expiry (T)**.

- **Call** → right to **buy**
- **Put** → right to **sell**
- The **option price (premium)** is the **amount you pay today** for that right.

Example:  
Buy a call on Apple (K = 100, premium = $5).  
If stock rises to $120 → payoff = 20, profit = 20 − 5 = 15.  
If it stays below 100 → option expires worthless, loss = 5.

So option price = **expected discounted payoff under risk-neutral world**.

---

### 🧮 Black–Scholes Model
Assumes stock follows Geometric Brownian Motion:

\[ dS = \mu S dt + \sigma S dW_t \]

Under risk-neutral measure (\( \mu → r \)):

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2)
\]

\[
d_1 = \frac{\ln(S_0/K) + (r + 0.5σ^2)T}{σ√T}, \quad d_2 = d_1 - σ√T
\]

Option price = discounted expected payoff = \( e^{-rT}E^Q[\text{Payoff}] \)

---

### 🧠 Risk-Neutral Probabilities
- **Concept:** Imagine a world where all assets grow at the risk-free rate (r).
- Used so that we can price assets **without arbitrage**:
  \[ E^Q[e^{-rT}S_T] = S_0 \]
- The “Q” measure makes pricing formulas simpler — all risk premia absorbed in probability weighting.

---

### 💡 Hedging Intuition
- Hold an option (nonlinear risk) and offset it by holding \( Δ = ∂C/∂S \) shares of stock.
- Portfolio becomes locally **risk-free** → must earn risk-free rate.
- This **no-arbitrage** argument leads directly to Black–Scholes PDE → fair pricing.

---

### 🔺 Gamma and Expiry
- **Gamma** = rate of change of Delta.
- Near expiry, small moves in stock drastically flip option value → **Gamma is high** near strike.
- Deep ITM/OTM options have low Gamma.

---

### 🧮 Monte Carlo Simulation Formula
To simulate under risk-neutral dynamics:

\[ S_T = S_0 e^{(r - 0.5σ^2)T + σ√T Z}, \quad Z∼N(0,1) \]

Why?
- This ensures \( E[S_T] = S_0 e^{rT} \) (risk-neutral drift).

Simulation steps:
1. Generate many random paths (Z).
2. Compute payoff = max(S_T − K, 0).
3. Average all payoffs → \( E^Q[\text{Payoff}] \).
4. Discount by \( e^{-rT} \) → price today.

---

### ⚖️ Put–Call Parity

Two portfolios that give identical payoffs must have equal prices:

\[ C + K e^{-rT} = P + S_0 \]

or equivalently,  
\[ C - P = S_0 - K e^{-rT} \]

This enforces **no-arbitrage** between calls, puts, and stock.

---

## 💼 2. Portfolio Optimization

### 🧠 Intuition
Allocate portfolio weights \( w = [w_1, ..., w_n] \) to balance **return vs risk**.

We want **maximum expected return per unit risk** → maximize **Sharpe ratio**.

---

### 🧮 Math Formulation

Expected portfolio return: \( R_p = w^T μ \)  
Portfolio variance: \( σ_p^2 = w^T Σ w \)

Sharpe ratio:
\[ S(w) = \frac{w^T μ - r_f}{\sqrt{w^T Σ w}} \]

Optimization problem:
\[
\max_{w} \frac{w^T μ - r_f}{\sqrt{w^T Σ w}} \quad
s.t. \quad \sum_i w_i = 1, \; w_i ≥ 0
\]

- **Decision variable:** weights \(w\)
- **Inputs:** expected returns (μ), covariance (Σ)
- Solver (like SLSQP) changes \(w\) to find max Sharpe ratio.

---

### 🧠 Intuitive Meaning
We are finding the best **combination of assets** that gives the **highest return per unit volatility**.  
It’s not about picking one best stock — it’s about finding the *best mix* given their correlations.

---

### ⚙️ Extensions
- **Min-variance portfolio:** minimize \( w^T Σ w \) subject to target return.
- **Efficient frontier:** curve of optimal risk–return combinations.
- **VaR / CVaR:** quantify downside risk beyond a confidence level.

---

### 🤖 Role of Machine Learning
ML was used to **predict expected returns (μ̂)** using historical features (momentum, volatility, etc.),  
and those predicted returns replaced historical means in the optimizer:

\[ w^* = \arg\max_w \frac{w^T μ̂}{\sqrt{w^T Σ w}} \]

→ ML = forecasting layer, Optimization = allocation layer.

---

## 📉 3. Pairs Trading – Statistical Arbitrage

### 💭 Intuition
Find two **cointegrated stocks** (move together long-term).  
When they diverge short-term, bet on **mean reversion**.

- Spread too high → short expensive, long cheap.  
- Spread too low → long expensive, short cheap.

Market-neutral: overall exposure ≈ 0.

---

### 🧮 Core Steps
1. Test for **cointegration**:
   \( P_1 = a + bP_2 + ε_t \), where \( ε_t \) stationary.
2. Define **spread** = \( P_1 - bP_2 \).
3. Compute **z-score** = \( (spread - μ)/σ \).
4. Trade when |z| > threshold, exit when |z| < small value.

Profits come from spread reverting.

---

### 🧠 Math–Intuition Link
Cointegration ensures **long-term equilibrium** → deviation is temporary noise.  
Z-score gives a **statistical measure** of mispricing magnitude.

---

## 🔗 How All Three Connect

| Component | Focus | Concept | Intuition |
|------------|--------|----------|------------|
| **Options Pricing** | Value of risk | Black–Scholes | What is the right price for future uncertainty? |
| **Portfolio Optimization** | Allocation | Markowitz / Sharpe Ratio | How do we balance risk vs return optimally? |
| **Pairs Trading** | Market-neutral profit | Cointegration & Mean Reversion | How do we exploit short-term inefficiencies? |

Machine Learning fits in as the *predictive* layer for returns,  
while options pricing and pairs trading handle *valuation* and *execution* logic.

---

### 🧠 Interview One-liner Summary

> “My project integrates three pillars of quantitative finance:  
> Black–Scholes for pricing risk, Markowitz optimization for allocating it, and cointegration-based pairs trading for exploiting market inefficiencies.  
> Machine Learning improves expected return forecasts, feeding into the optimizer to enhance Sharpe ratios.  
> Together, it forms an end-to-end quantitative framework that prices, allocates, and trades risk efficiently.”

---
