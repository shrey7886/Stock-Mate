"""
Financial knowledge base for StockMate RAG.
Covers Indian market rules, portfolio theory, tax, SIP, and investing concepts.
Each chunk is independently retrievable by semantic similarity.
"""
from __future__ import annotations

# Each chunk: id, topic, title, content, tags
KNOWLEDGE_CHUNKS: list[dict] = [

    # ── TAXATION ─────────────────────────────────────────────────────────────

    {
        "id": "tax_ltcg_equity",
        "topic": "tax",
        "title": "LTCG on Equity in India (Budget 2024)",
        "content": (
            "Long-Term Capital Gains (LTCG) on listed equity shares and equity mutual funds "
            "in India: if you hold equity for MORE than 12 months, gains are classified as LTCG. "
            "Post Budget 2024, LTCG is taxed at 12.5% (raised from 10%). "
            "Importantly, the first ₹1.25 lakh of LTCG per financial year is EXEMPT from tax "
            "(previously ₹1 lakh). So if your total long-term gains are ₹2 lakh, you pay 12.5% "
            "only on ₹75,000 (i.e., ₹2L minus ₹1.25L exemption). "
            "No indexation benefit is available for equity LTCG. "
            "Pro tax-planning tip: if your unrealized LTCG is approaching ₹1.25L, consider "
            "'harvesting' by selling and rebooking before March 31 each year to reset the base."
        ),
        "tags": ["tax", "ltcg", "capital gains", "equity", "budget 2024", "long term"],
    },
    {
        "id": "tax_stcg_equity",
        "topic": "tax",
        "title": "STCG on Equity in India (Budget 2024)",
        "content": (
            "Short-Term Capital Gains (STCG) on listed equity shares: if you sell within 12 months "
            "of buying, the profit is STCG — taxed at a flat 20% (Budget 2024 raised this from 15%). "
            "STCG applies regardless of your income tax slab. "
            "Example: you bought 100 shares of INFY at ₹1,400 and sold at ₹1,600 after 8 months. "
            "Gain = ₹20,000. STCG tax = ₹20,000 × 20% = ₹4,000. "
            "Key point: STT (Securities Transaction Tax) is already deducted at source when you trade, "
            "but you still owe income tax on the profit. Always plan your trade horizon with these "
            "rates in mind — staying invested even one extra day beyond 12 months can halve your tax rate."
        ),
        "tags": ["tax", "stcg", "capital gains", "short term", "equity", "budget 2024"],
    },
    {
        "id": "tax_loss_harvesting",
        "topic": "tax",
        "title": "Tax-Loss Harvesting Strategy",
        "content": (
            "Tax-loss harvesting means deliberately selling a loss-making position to book a capital "
            "loss that offsets your capital gains, reducing your tax bill. "
            "Rules in India: short-term capital losses (STCL) can be set off against both STCG and LTCG. "
            "Long-term capital losses (LTCL) can ONLY be set off against LTCG. "
            "Unabsorbed capital losses can be carried forward for 8 financial years. "
            "Example: You have ₹50,000 LTCG from HDFC Bank. You also hold Zomato with ₹30,000 unrealized LTCL. "
            "By selling Zomato, your net taxable LTCG drops to ₹20,000 (below the ₹1.25L exemption — tax = ₹0). "
            "Caution: selling a position purely for tax purposes means you exit that trade. "
            "If you still like the stock, you can re-enter after 30 days to avoid wash-sale -style scrutiny "
            "(India doesn't have a strict wash-sale rule like the US, but reinvesting immediately defeats the purpose)."
        ),
        "tags": ["tax", "tax loss harvesting", "capital loss", "stcl", "ltcl", "tax planning"],
    },
    {
        "id": "tax_stt_securities",
        "topic": "tax",
        "title": "STT — Securities Transaction Tax",
        "content": (
            "Securities Transaction Tax (STT) is a tax levied at the time of buying/selling equities on "
            "Indian exchanges (NSE/BSE). "
            "Current rates (2024): Equity delivery buy = 0.1%, delivery sell = 0.1%. "
            "Intraday (MIS) buy = nil, intraday sell = 0.025%. "
            "Futures sell = 0.02%. Options sell = 0.1% of premium. "
            "STT is automatically deducted by your broker — it appears in your contract note. "
            "It is NOT deductible against capital gains, but for traders, it can be claimed as a business expense. "
            "Budget 2024 increased STT on F&O significantly to curb retail speculation. "
            "For long-term equity investors with delivery trades, STT impact is minor (0.1% each way)."
        ),
        "tags": ["tax", "stt", "securities transaction tax", "equity", "trading"],
    },
    {
        "id": "tax_dividend_income",
        "topic": "tax",
        "title": "Dividend Tax in India",
        "content": (
            "Since FY2020-21, dividends from Indian companies are fully taxable in the hands of the investor "
            "at their applicable income tax slab rate. The old DDT (Dividend Distribution Tax) system was abolished. "
            "TDS: If dividend income from a single company exceeds ₹5,000 in a year, the company deducts TDS at 10%. "
            "For NRIs, TDS is 20%. You can claim this TDS credit when filing your ITR. "
            "Practical implication: If you're in the 30% tax bracket, ₹10,000 in dividends costs you ₹3,000 in tax. "
            "Growth-option mutual funds are more tax-efficient than dividend-payout plans for high-bracket investors. "
            "Dividend income must be reported under 'Income from Other Sources' in ITR."
        ),
        "tags": ["tax", "dividend", "tds", "income tax", "mutual fund"],
    },

    # ── PORTFOLIO MANAGEMENT ─────────────────────────────────────────────────

    {
        "id": "portfolio_diversification",
        "topic": "portfolio",
        "title": "Diversification: The Only Free Lunch in Investing",
        "content": (
            "Diversification is the practice of spreading investments across different assets, sectors, "
            "and geographies to reduce unsystematic (company-specific) risk. "
            "A well-diversified Indian equity portfolio typically has: "
            "12–20 stocks across 5+ sectors, no single stock exceeding 10–15% of portfolio, "
            "exposure to large-cap (stability), mid-cap (growth), and possibly small-cap (high risk/reward). "
            "The math: two perfectly correlated assets offer NO diversification benefit. "
            "Two assets with correlation < 1 reduce portfolio volatility even if both are individually risky. "
            "Common mistake: holding 20 PSU bank stocks is NOT diversification — they're all highly correlated. "
            "True diversification = different industries that move independently (pharma + IT + FMCG + infra). "
            "Over-diversification (50+ stocks) dilutes alpha and makes portfolio management unmanageable."
        ),
        "tags": ["diversification", "portfolio", "risk", "sectors", "correlation"],
    },
    {
        "id": "portfolio_rebalancing",
        "topic": "portfolio",
        "title": "Portfolio Rebalancing — When and How",
        "content": (
            "Rebalancing means bringing your portfolio back to its intended asset/stock allocation after "
            "market movements cause drift. "
            "Example: You planned 60% large-cap, 40% mid-cap. After a mid-cap rally, it's now 50/50. "
            "Rebalancing = sell some mid-cap, buy large-cap to restore 60/40. "
            "When to rebalance: "
            "1. Calendar-based: Quarterly or annually (simple, low-effort). "
            "2. Threshold-based: When any allocation drifts >5% from target (more responsive). "
            "3. Event-based: After major market events (crash, bubble). "
            "Tax considerations: Selling to rebalance triggers capital gains. Consider using NEW money "
            "(SIP contributions) to buy underweighted assets instead of selling — avoids tax. "
            "In tax-advantaged accounts or for LTCG-eligible holdings, selling is more tax-efficient."
        ),
        "tags": ["rebalancing", "portfolio", "asset allocation", "drift", "tax"],
    },
    {
        "id": "portfolio_concentration_risk",
        "topic": "portfolio",
        "title": "Concentration Risk — When One Stock is Too Much",
        "content": (
            "Concentration risk is the danger of holding too large a position in a single stock, sector, "
            "or asset class. "
            "Rule of thumb: No single stock should exceed 10–15% of your total equity portfolio. "
            "If one stock is >30%, you have significant concentration risk. "
            "Real examples: Investors heavily concentrated in Yes Bank (2020), DHFL, or IL&FS suffered "
            "catastrophic losses because all eggs were in one basket. "
            "Even great companies can underperform for years (IT sector 2022, PSU banks 2015–2019). "
            "Signs you're over-concentrated: one stock's daily move determines whether you're happy or sad. "
            "Solution: Trim the oversized position systematically — don't sell all at once (tax impact + "
            "you might be wrong about the timing). Sell 20–30% of oversized portion quarterly."
        ),
        "tags": ["concentration", "risk", "single stock", "portfolio", "diversification"],
    },
    {
        "id": "portfolio_drawdown",
        "topic": "portfolio",
        "title": "Understanding Portfolio Drawdown",
        "content": (
            "Drawdown = the decline from a portfolio's peak value to its current trough, expressed as %. "
            "Formula: Drawdown% = (Current Value - Peak Value) / Peak Value × 100. "
            "A 20% drawdown requires a 25% recovery just to break even. "
            "A 50% drawdown requires a 100% recovery. This asymmetry is why limiting drawdowns matters. "
            "Max drawdown benchmarks for Indian equity: "
            "Large-cap during bad bear market: 30–40% (Nifty fell ~60% in 2008-09). "
            "Mid/small-cap: can fall 50–70% in severe bear markets. "
            "Portfolio at -10%: concerning if recent, normal if multi-year. "
            "Portfolio at -20%+: serious — review allocation, check if thesis still holds. "
            "Counter-intuitive truth: If you're a long-term investor still in accumulation phase, "
            "drawdowns are your best friend — they let you buy more at lower prices via SIP."
        ),
        "tags": ["drawdown", "risk", "portfolio", "recovery", "loss"],
    },
    {
        "id": "portfolio_asset_allocation",
        "topic": "portfolio",
        "title": "Asset Allocation — The Most Important Decision",
        "content": (
            "Asset allocation determines >90% of your portfolio's long-term returns and risk. "
            "Core asset classes for Indian investors: "
            "1. Equity (stocks/equity MF): Highest return potential, highest volatility. "
            "2. Debt (bonds, FD, debt MF): Steady income, lower risk, inflation risk. "
            "3. Gold (via ETF like GOLDBEES, SGBs, or Gold MF): Inflation hedge, currency hedge. "
            "4. Real Estate (REITs or physical): Illiquid but inflation-hedging. "
            "5. International equity: Diversification from INR and Indian market cycles. "
            "Simple age-based rule: % in equity = 100 minus your age. "
            "A 25-year-old: 75% equity, 25% debt/gold. "
            "A 50-year-old: 50% equity, 50% debt/gold. "
            "More sophisticated allocation considers risk tolerance, time horizon, and income stability."
        ),
        "tags": ["asset allocation", "equity", "debt", "gold", "portfolio", "diversification"],
    },

    # ── SIP & INVESTMENT STRATEGIES ──────────────────────────────────────────

    {
        "id": "sip_basics",
        "topic": "sip",
        "title": "SIP — Systematic Investment Plan Explained",
        "content": (
            "A Systematic Investment Plan (SIP) means investing a fixed amount at regular intervals "
            "(monthly, weekly) into an equity or mutual fund, regardless of market levels. "
            "SIP is India's most popular retail investing mechanism — ₹20,000+ crore flows in via SIP monthly. "
            "Key benefits: "
            "1. Rupee Cost Averaging: You buy more units when prices are low, fewer when high. "
            "   Over time, your average cost per unit is lower than the average market price. "
            "2. Discipline: Removes the need to 'time the market' (which almost nobody can do consistently). "
            "3. Compounding: Regular investing lets compounding work from early on. "
            "SIP in direct stocks: You can also run a 'stock SIP' by buying fixed ₹ amount in a stock monthly. "
            "Best SIP candidates: market leaders with strong moats (you're comfortable holding for 5–10 years). "
            "Avoid SIP in highly cyclical or loss-making businesses — SIP amplifies mistakes too."
        ),
        "tags": ["sip", "systematic investment plan", "mutual fund", "rupee cost averaging", "investing"],
    },
    {
        "id": "sip_power_of_compounding",
        "topic": "sip",
        "title": "Power of Compounding with SIP",
        "content": (
            "Compounding = earning returns on your returns. Einstein allegedly called it the 8th wonder, "
            "and your portfolio will agree after 20 years. "
            "SIP compounding example (12% annualized return): "
            "₹5,000/month for 10 years: Invested ₹6L → Grows to ~₹11.6L (1.9x). "
            "₹5,000/month for 20 years: Invested ₹12L → Grows to ~₹49.9L (4.1x). "
            "₹5,000/month for 30 years: Invested ₹18L → Grows to ~₹1.76 crore (9.8x). "
            "The jump from 20→30 years adds only ₹6L invested but ₹1.26 crore in wealth. "
            "That's the power of the final years of compounding. "
            "Key insight: TIME in the market > TIMING the market. "
            "Starting 10 years earlier with half the SIP amount often beats starting later with double the amount. "
            "The best time to start a SIP was 10 years ago. The second best time is today."
        ),
        "tags": ["sip", "compounding", "wealth creation", "long term", "investment"],
    },
    {
        "id": "sip_calculation_method",
        "topic": "sip",
        "title": "How to Calculate SIP Needed for a Goal",
        "content": (
            "To calculate the monthly SIP needed to reach a target corpus: "
            "Use the Future Value of annuity formula: "
            "FV = PMT × [((1+r)^n - 1) / r] × (1+r) "
            "Where: FV = target corpus, PMT = monthly SIP, r = monthly return rate (annual/12), "
            "n = number of months. "
            "Rearranging: PMT = FV × r / (((1+r)^n - 1) × (1+r)) "
            "Example: Goal = ₹1 crore in 15 years, assuming 12% annual return. "
            "r = 12/100/12 = 0.01, n = 180 "
            "PMT = 1,00,00,000 × 0.01 / (((1.01)^180 - 1) × 1.01) ≈ ₹18,800/month. "
            "Important: 'Expected return' is an assumption, not a guarantee. "
            "12% is a common assumption for diversified Indian equity over the long run, but actual returns vary. "
            "Always set aside a buffer — target 110–120% of your goal amount to account for variability."
        ),
        "tags": ["sip", "goal", "calculation", "corpus", "target", "formula"],
    },

    # ── INDIAN MARKET STRUCTURE ───────────────────────────────────────────────

    {
        "id": "market_nifty_sensex",
        "topic": "market",
        "title": "Nifty 50 and Sensex — India's Key Indices",
        "content": (
            "Nifty 50 (NSE) and Sensex (BSE) are India's two headline benchmark indices. "
            "Nifty 50: Top 50 companies by free-float market cap traded on NSE. Rebalanced semi-annually. "
            "Sectoral composition (approx. 2024): Financial Services ~33%, IT ~13%, Oil&Gas ~12%, "
            "Consumer goods ~9%, Auto ~7%, Pharma ~5%, others. "
            "Sensex: Top 30 stocks on BSE — heavily overlaps Nifty. "
            "When people say 'market is up 1%', they usually mean Nifty 50. "
            "Key sector indices: Nifty Bank, Nifty IT, Nifty Pharma, Nifty Auto, Nifty FMCG. "
            "Bull market: Nifty > 200-day moving average, fear index (VIX) low. "
            "Bear market: Nifty < 200 DMA, VIX elevated, FIIs selling. "
            "India VIX: measures expected volatility of Nifty 50 over next 30 days. VIX > 20 = high fear."
        ),
        "tags": ["nifty", "sensex", "index", "market", "bse", "nse", "benchmark"],
    },
    {
        "id": "market_fii_dii",
        "topic": "market",
        "title": "FIIs and DIIs — Market Movers",
        "content": (
            "FII (Foreign Institutional Investors) and DII (Domestic Institutional Investors) are the "
            "two largest participants in Indian equity markets. "
            "FIIs include foreign hedge funds, pension funds, sovereign wealth funds. "
            "When FIIs sell (net outflows), markets typically fall — they control ~23% of Nifty free float. "
            "DIIs include Indian mutual funds (AMCs), insurance companies (LIC), banks. "
            "DIIs have increasingly absorbed FII selling thanks to retail SIP inflows (₹20,000+ cr/month). "
            "Net FII data is published daily on NSE website — it's one of the most watched numbers. "
            "USDINR matters: When rupee weakens, FIIs lose more in dollar terms, triggering further selling. "
            "DII buying often supports markets at 200 DMA — they see corrections as buying opportunities. "
            "Practical use: Large FII selling during global risk-off is often a buying opportunity for "
            "long-term investors, not a reason to panic."
        ),
        "tags": ["fii", "dii", "institutional", "market", "flows", "foreign investor"],
    },
    {
        "id": "market_etf_goldbees",
        "topic": "market",
        "title": "ETFs in India — Gold ETF, Index ETF",
        "content": (
            "Exchange-Traded Funds (ETFs) are funds that trade on stock exchanges like regular shares. "
            "Popular Indian ETFs: "
            "GOLDBEES (Nippon India Gold BeES): Tracks domestic gold prices. 1 unit ≈ 0.01g of gold. "
            "A tax-efficient alternative to buying physical gold. LTCG applies (>3 years = better tax). "
            "Actually for gold ETFs, holding period for LTCG is 24 months (not 12 months). "
            "NIFTYBEES, NETF, JUNIORBEES: Track Nifty 50 or Nifty Next 50. "
            "Bank Nifty ETFs: BANKBEES, ICICIBANKN. "
            "Advantages of ETFs: low expense ratio (0.05–0.20%), intraday liquidity, no fund manager risk. "
            "Disadvantages: tracking error (especially for illiquid ETFs), bid-ask spread. "
            "For most retail investors, a simple Nifty 50 ETF SIP outperforms most active funds over 10+ years "
            "after expense ratios — this is the 'Bogleheads' philosophy applied to India."
        ),
        "tags": ["etf", "gold", "goldbees", "nifty", "index fund", "passive investing"],
    },
    {
        "id": "market_circuit_breakers",
        "topic": "market",
        "title": "Market Circuit Breakers and Upper/Lower Circuits",
        "content": (
            "SEBI mandates circuit breakers (CBs) to curb extreme volatility in Indian markets. "
            "Index-level CBs (Nifty/Sensex): "
            "10% fall: 45-min halt (after first breach). "
            "15% fall: 2-hour halt. "
            "20% fall: Market closes for the day. "
            "Stock-level circuits: Each stock has daily price bands (5%, 10%, or 20%). "
            "If a stock hits its upper circuit (UC), only buyers exist — cannot buy more that day. "
            "If it hits lower circuit (LC), only sellers exist — cannot exit if you want to. "
            "Lower circuits are a serious liquidity risk for small/mid-cap stocks. "
            "Always check a stock's circuit filter before investing large amounts. "
            "Stocks under SEBI ASM (Additional Surveillance Mechanism) or GSM (Graded Surveillance) "
            "have tighter circuits and increased surveillance due to unusual trading patterns."
        ),
        "tags": ["circuit breaker", "upper circuit", "lower circuit", "sebi", "liquidity risk"],
    },
    {
        "id": "market_sebi_basics",
        "topic": "market",
        "title": "SEBI — The Market Regulator",
        "content": (
            "SEBI (Securities and Exchange Board of India) is the apex regulator of Indian capital markets. "
            "Established 1992. Headquarters: Mumbai. "
            "Key SEBI functions: Regulates stock exchanges (NSE, BSE), listed companies, brokers, "
            "depositories (CDSL, NSDL), mutual funds, and FIIs. "
            "Important SEBI rules for investors: "
            "T+1 settlement: From Jan 2023, most equity trades settle in T+1 (next trading day). "
            "Demat accounts mandatory for listed shares. "
            "ASBA (Applications Supported by Blocked Amount): For IPO applications. "
            "Nominee registration mandatory from 2024. "
            "SEBI's investor protection measures include the SCORES portal for lodging complaints. "
            "If your broker overcharges or mishandles funds, SEBI SCORES is your first avenue."
        ),
        "tags": ["sebi", "regulator", "compliance", "investor protection", "indian market"],
    },

    # ── RISK & TECHNICAL ANALYSIS ─────────────────────────────────────────────

    {
        "id": "risk_beta",
        "topic": "risk",
        "title": "Beta — Measuring Stock Volatility vs Market",
        "content": (
            "Beta measures a stock's sensitivity to market movements relative to a benchmark (usually Nifty 50). "
            "Beta = 1.0: Stock moves in line with the market. "
            "Beta > 1.0: More volatile than market (e.g., Beta 1.5 = if Nifty rises 10%, stock rises 15%). "
            "Beta < 1.0: Less volatile (e.g., Beta 0.5 = if Nifty falls 10%, stock falls only 5%). "
            "Beta < 0: Moves opposite to market (rare — gold miners sometimes). "
            "High-beta sectors in India: IT, Metal, Real Estate, Mid-cap. "
            "Low-beta sectors: FMCG, Utilities, Pharma. "
            "Important: Beta is backward-looking (calculated from historical data). "
            "A low-beta stock can still crash on company-specific news. "
            "Beta is most useful in a well-diversified portfolio — a single stock's beta is less meaningful "
            "in a concentrated portfolio."
        ),
        "tags": ["beta", "volatility", "risk", "market", "nifty"],
    },
    {
        "id": "risk_sharpe_ratio",
        "topic": "risk",
        "title": "Sharpe Ratio — Risk-Adjusted Returns",
        "content": (
            "Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Standard Deviation of Returns. "
            "It measures how much extra return you earn per unit of risk taken. "
            "Risk-free rate in India: typically 6–7% (10-year G-Sec yield). "
            "If your portfolio returns 14% with 15% standard deviation: "
            "Sharpe = (14% - 7%) / 15% = 0.47. "
            "Generally: Sharpe > 1.0 = good, > 2.0 = very good. "
            "A portfolio returning 12% with low volatility (Sharpe 0.8) is often BETTER than "
            "one returning 18% with massive swings (Sharpe 0.5). "
            "Mutual funds report Sharpe ratios — compare within the same category. "
            "Common trap: High recent returns usually come with high volatility. "
            "Check Sharpe over 3–5 years, not just 1 year. "
            "A fund with consistent 13% returns and low drawdowns beats one with hot/cold 20% or 5% swings."
        ),
        "tags": ["sharpe ratio", "risk-adjusted return", "volatility", "portfolio", "mutual fund"],
    },
    {
        "id": "technical_rsi",
        "topic": "technical",
        "title": "RSI — Relative Strength Index",
        "content": (
            "RSI (Relative Strength Index) is a momentum oscillator measured 0–100. "
            "Formula: RSI = 100 - [100 / (1 + Average Gain / Average Loss)] over 14 periods. "
            "Interpretation: "
            "RSI > 70: Overbought — stock may be due for a pullback. Consider cautious/hold stance. "
            "RSI < 30: Oversold — stock may be due for a bounce. Potential buy opportunity. "
            "RSI 40–60: Neutral zone. "
            "In strong bull trends, RSI can remain overbought (70+) for weeks — 'overbought can stay overbought'. "
            "In bear trends, RSI rarely recovers to 70 and stays in 20–50 range. "
            "RSI divergence: Stock makes new high but RSI doesn't = bearish divergence (sell signal). "
            "Stock makes new low but RSI doesn't = bullish divergence (buy signal). "
            "Best used ALONGSIDE other indicators (Support/Resistance, volume) — not as a standalone signal."
        ),
        "tags": ["rsi", "technical analysis", "momentum", "overbought", "oversold"],
    },
    {
        "id": "technical_moving_averages",
        "topic": "technical",
        "title": "Moving Averages — Trend Following Tools",
        "content": (
            "Moving Averages (MA) smooth out price data to identify trends. "
            "Types: Simple MA (SMA) = average of last N prices. Exponential MA (EMA) = weighted toward recent prices. "
            "Key levels traders watch in Indian markets: "
            "20 DMA (Daily Moving Average): Short-term trend. Price above 20D = short-term bullish. "
            "50 DMA: Medium-term trend. "
            "200 DMA: Long-term trend. Price above 200D = in bull market. "
            "Golden Cross: 50 DMA crosses ABOVE 200 DMA = strong bullish signal. "
            "Death Cross: 50 DMA crosses BELOW 200 DMA = bearish signal. "
            "Price below 200 DMA: Usually means the stock is in a downtrend. "
            "Important caveat: MAs are LAGGING indicators. By the time a Golden Cross confirms, "
            "you may have already missed 10–15% of the move. "
            "MAs work better for long-term investors as trend filters, not as precise entry/exit signals."
        ),
        "tags": ["moving average", "200 dma", "golden cross", "trend", "technical analysis"],
    },
    {
        "id": "technical_support_resistance",
        "topic": "technical",
        "title": "Support and Resistance Levels",
        "content": (
            "Support = a price level where buying interest is strong enough to stop further decline. "
            "Resistance = a price level where selling pressure is strong enough to stop further rise. "
            "Why they work: price memory (traders remember previous highs/lows) and psychological round numbers. "
            "Previous highs become resistance; when broken, they flip to support. "
            "Example: Nifty struggled at 18,000 for months. Once broken, 18,000 became strong support. "
            "For individual stocks, watch: 52-week highs/lows, previous all-time highs, price gaps. "
            "Volume confirmation: A breakout above resistance WITH high volume is more reliable. "
            "A breakout on low volume often 'fakeouts' and reverses. "
            "For long-term investors: Use support levels to add to positions (buy near support). "
            "Use resistance levels as partial profit-booking targets. "
            "Don't rely purely on technical levels — fundamental thesis must also hold."
        ),
        "tags": ["support", "resistance", "technical analysis", "price", "breakout"],
    },

    # ── FUNDAMENTAL ANALYSIS ──────────────────────────────────────────────────

    {
        "id": "fundamental_pe_ratio",
        "topic": "fundamental",
        "title": "P/E Ratio — Price to Earnings Explained",
        "content": (
            "P/E Ratio = Stock Price / Earnings Per Share (EPS). "
            "It answers: 'How many years of earnings am I paying for at current price?' "
            "High P/E (e.g., 40x): Market expects high future growth (true for tech, new-age companies). "
            "Low P/E (e.g., 8x): Cheap OR market expects slow/declining growth (value trap risk). "
            "Average Nifty 50 P/E historically: 18–22x. Above 25x = expensive historically. "
            "Sector P/E varies wildly: "
            "FMCG: 40–60x (stable, premium), PSU banks: 5–10x (high NPA concerns, discount). "
            "IT: 25–30x (growth premium), Infra: 10–15x. "
            "P/E alone is useless without growth context — use PEG ratio (P/E / Growth Rate). "
            "PEG < 1.0 = potentially undervalued relative to growth. "
            "Forward P/E (based on next year's estimated earnings) is more relevant than trailing P/E. "
            "Never compare P/E across different industries — it's like comparing apples to airplanes."
        ),
        "tags": ["pe ratio", "valuation", "fundamental analysis", "earnings", "PEG"],
    },
    {
        "id": "fundamental_debt_equity",
        "topic": "fundamental",
        "title": "Debt-to-Equity Ratio — Financial Health Indicator",
        "content": (
            "D/E Ratio = Total Debt / Shareholders' Equity. "
            "High D/E means company is heavily leveraged — good in low interest rate environment, "
            "dangerous when rates rise (as seen globally in 2022-23). "
            "Industry-specific benchmarks: "
            "Capital-intensive industries (steel, infra, telecom): D/E of 1–3x is normal. "
            "Asset-light businesses (IT, FMCG): D/E should be near 0. "
            "Banks/NBFCs: Use different leverage metrics (CAR, NIM, GNPA%). "
            "Red flags: D/E > 2x for a non-capital-intensive company, "
            "interest coverage ratio < 2x (profit barely covers interest payments), "
            "debt growing faster than revenue. "
            "For retail investors, simple rule: Prefer companies with net debt-free balance sheets "
            "for their core holdings — especially in volatile sectors. "
            "Companies with zero debt can invest aggressively when markets crash, gaining market share."
        ),
        "tags": ["debt", "equity", "leverage", "balance sheet", "fundamental analysis"],
    },
    {
        "id": "fundamental_roe_roce",
        "topic": "fundamental",
        "title": "ROE and ROCE — Profitability Efficiency Metrics",
        "content": (
            "ROE (Return on Equity) = Net Profit / Shareholders' Equity × 100%. "
            "Measures how efficiently management uses shareholder money. "
            "ROE > 15–20% consistently = good business. ROE > 25% = excellent. "
            "ROCE (Return on Capital Employed) = EBIT / (Total Assets - Current Liabilities) × 100%. "
            "Better than ROE because it considers ALL capital (debt + equity). "
            "ROCE > 15% generally good. ROCE > WACC (weighted avg cost of capital) = value creation. "
            "The power combo: High ROE + High ROCE + Low Debt = a company with genuine competitive advantage. "
            "Classic examples: HDFC Bank, Asian Paints, Pidilite — all have consistently high ROE/ROCE "
            "over decades with low/nil debt. "
            "Warning: High ROE can be manufactured by taking on debt (inflates E denominator's complement). "
            "Always cross-check ROE with D/E ratio."
        ),
        "tags": ["roe", "roce", "return on equity", "profitability", "fundamental analysis"],
    },

    # ── COMMON INVESTOR MISTAKES ──────────────────────────────────────────────

    {
        "id": "mistakes_panic_selling",
        "topic": "behavior",
        "title": "Panic Selling — The Most Expensive Mistake",
        "content": (
            "Panic selling = liquidating investments during a market crash out of fear, usually at the worst time. "
            "Data: Missing the 10 best trading days in a year dramatically reduces annual returns. "
            "Nifty historical: investors who stayed invested through 2008, 2016, 2020 crashes "
            "were rewarded handsomely within 12–18 months. "
            "Psychology behind it: Loss aversion — humans feel losses ~2x more intensely than equivalent gains. "
            "News during crashes always sounds apocalyptic — this is normal, not a signal to exit. "
            "When to actually sell: If your investment THESIS has changed, not just the price. "
            "E.g., a bank with rising NPAs despite broad rally (thesis broken) → review position. "
            "But if a quality company fell 30% in a broad market correction → thesis intact → hold or add. "
            "The best investors do nothing during crashes. The worst investors sell at the bottom and "
            "wait until new all-time highs before re-entering."
        ),
        "tags": ["panic selling", "behavior", "investor mistakes", "correction", "crash"],
    },
    {
        "id": "mistakes_averaging_down",
        "topic": "behavior",
        "title": "Averaging Down — Smart or Dangerous?",
        "content": (
            "Averaging down = buying more of a stock as its price falls to reduce your average cost. "
            "Warren Buffett does it. Most retail investors do it wrong. The difference: "
            "When it's smart: The stock fell due to broad market weakness, not company fundamentals. "
            "You have high conviction in the business after re-checking the thesis. "
            "You're averaging into strong, profitable companies (not turnaround stories). "
            "When it's dangerous — 'catching a falling knife': "
            "The company has fundamental problems (fraud, structural disruption, debt crisis). "
            "You're averaging down in hope ('it can't fall more'), not conviction. "
            "You're using leverage or emergency funds to average down. "
            "Golden rule: Before averaging down, ask — 'If I didn't already own this stock, "
            "would I buy it fresh at this price?' If no → don't average. If yes → averaging may be valid. "
            "A useful heuristic: Maximum 3 averaging-down tranches per stock. If down after 3 tranches, "
            "the thesis may be fundamentally broken."
        ),
        "tags": ["averaging down", "cost averaging", "behavior", "investor mistakes"],
    },
    {
        "id": "mistakes_fomo",
        "topic": "behavior",
        "title": "FOMO Investing — Fear of Missing Out",
        "content": (
            "FOMO investing = buying a stock/sector only because it has already gone up a lot "
            "and you don't want to 'miss out' on further gains. "
            "Classic FOMO scenarios in India: "
            "Railway stocks (2024): 10–15x returns, then massive correction. "
            "PSU stocks (2022-23): Sudden hype, many retail investors bought near peaks. "
            "New-age tech IPOs (Zomato, Paytm, Nykaa 2021): Massive FOMO, crashed 60-80%. "
            "SME IPOs with 100x subscription — often frenzy, not fundamentals. "
            "How to avoid FOMO: "
            "1. Ask: What is the business? What is the valuation? Does it make money? "
            "2. If you can't explain why the stock should be worth more in 5 years, don't buy. "
            "3. If your reason to buy is 'everyone is buying it' or 'it's been going up' — that's FOMO. "
            "4. There's ALWAYS another opportunity. Markets never run out of good stocks."
        ),
        "tags": ["fomo", "behavior", "investor mistakes", "momentum", "ipo", "speculation"],
    },
    {
        "id": "mistakes_over_trading",
        "topic": "behavior",
        "title": "Over-Trading — Churning Kills Returns",
        "content": (
            "Over-trading = buying and selling too frequently, generating excessive transaction costs and taxes. "
            "Impact on returns: "
            "Every buy + sell in equity delivery = 0.2% STT + brokerage (~0.01–0.03%) + DP charges. "
            "Annual churn of 100% of portfolio costs ~0.5–1% in transaction costs alone. "
            "Plus short-term capital gains tax (20%) on profits. "
            "A 20% gain on a ₹1L position = ₹20,000. After 20% STCG tax = ₹16,000 net. "
            "If you had held 12+ more months, LTCG would be ₹10,000 tax (50% less). "
            "Dalbar study (US, but applies globally): Average retail investor significantly underperforms "
            "the index due to bad timing and over-trading. "
            "Simple rule: Unless your thesis has changed, don't trade. "
            "Good portfolio management is mostly inactivity, with occasional decisive action."
        ),
        "tags": ["over-trading", "transaction costs", "behavior", "tax", "holding period"],
    },

    # ── GOLD & ALTERNATIVE INVESTMENTS ───────────────────────────────────────

    {
        "id": "gold_investment_india",
        "topic": "gold",
        "title": "Gold as an Investment in India",
        "content": (
            "Gold has traditionally served as a hedge against inflation, currency depreciation, and "
            "geopolitical uncertainty in India. "
            "Ways to invest in gold: "
            "1. Physical gold: High purity concerns, storage/insurance cost, making charges on exit. "
            "   Generally NOT recommended for investment purposes. "
            "2. Gold ETF (e.g., GOLDBEES): Stock exchange-traded, 99.5% purity, demat form. "
            "   Expense ratio ~0.5%. Easy to buy/sell in demat. "
            "3. Sovereign Gold Bond (SGB): RBI-issued, 2.5% annual interest on top of gold price appreciation. "
            "   8-year tenure, but can sell on exchange after 5 years. ZERO capital gains tax if held to maturity. "
            "   Best option for long-term gold exposure. Note: RBI may have paused new SGB issuances in 2024. "
            "4. Gold Mutual Funds (invest in Gold ETFs): For investors without demat accounts. "
            "Tax: Gold ETF/MF: LTCG after 24 months at 12.5% flat. SGB: Tax-free at maturity. "
            "Ideal gold allocation: 5–15% of total portfolio as a hedge. Not a primary return driver."
        ),
        "tags": ["gold", "goldbees", "sgb", "sovereign gold bond", "hedge", "inflation"],
    },

    # ── MUTUAL FUNDS ──────────────────────────────────────────────────────────

    {
        "id": "mutual_fund_types",
        "topic": "mutual_fund",
        "title": "Types of Mutual Funds in India",
        "content": (
            "Indian mutual funds categorized by SEBI (SEBI circular Oct 2017): "
            "Equity funds: "
            "  Large-cap: Invest ≥80% in top 100 companies by market cap. Stable. "
            "  Mid-cap: ≥65% in 101–250th companies. Higher risk/return. "
            "  Small-cap: ≥65% in 251st company onward. Highest risk/return. "
            "  Flexi-cap: No market-cap restriction. Fund manager decides. "
            "  ELSS (Equity Linked Savings Scheme): Tax-saving, 3-year lock-in, ₹1.5L deduction under 80C. "
            "Debt funds: "
            "  Liquid funds: Very short maturity. Parking money. Better than savings account. "
            "  Short-duration, Medium-duration, Long-duration based on portfolio duration. "
            "Hybrid funds: Mix of equity and debt. Balanced Advantage Funds (BAF) adjust allocation dynamically. "
            "  BAF/Dynamic Asset Allocation: Popular for conservative investors. "
            "Index funds / ETFs: Passive, track Nifty/Sensex. Low expense ratio (0.1–0.2%). "
            "Fund of Funds (FoF): Invest in other funds, including international funds."
        ),
        "tags": ["mutual fund", "large cap", "mid cap", "small cap", "elss", "index fund"],
    },
    {
        "id": "mutual_fund_direct_vs_regular",
        "topic": "mutual_fund",
        "title": "Direct vs Regular Mutual Fund Plans",
        "content": (
            "Direct plan: Invest directly with the AMC (Asset Management Company) without a distributor/broker. "
            "Lower expense ratio by 0.5–1.25% compared to regular plans. "
            "Regular plan: Sold through distributors/brokers who earn a commission (trail commission). "
            "This commission comes from YOUR returns. "
            "Long-term impact of 1% extra expense ratio: "
            "₹10,000/month SIP for 20 years at 12% (direct) = ~₹97L. "
            "Same SIP at 11% (regular, 1% higher expense) = ~₹85L. "
            "Difference: ₹12L — just from the expense ratio! "
            "How to invest in direct plans: MF utility, AMC websites, or apps like Zerodha Coin, MF Central, "
            "Groww Direct, Kuvera, INDmoney. "
            "Caveat: Regular plans justify their cost ONLY if the advisor provides genuine financial planning "
            "value beyond just selling products. Pure product sellers with no advice = pay for direct plan."
        ),
        "tags": ["direct plan", "regular plan", "mutual fund", "expense ratio", "returns"],
    },

    # ── INTRADAY & TRADING ────────────────────────────────────────────────────

    {
        "id": "intraday_risks",
        "topic": "trading",
        "title": "Intraday Trading — Risks and Reality",
        "content": (
            "Intraday trading = buying and selling within the same trading day (9:15 AM – 3:30 PM IST). "
            "Position must be squared off by 3:15 PM or broker auto-squares at market price. "
            "SEBI study (2022–23): 91% of individual intraday F&O traders lost money. "
            "Average net loss per trader was ₹1.1 lakh. "
            "Why most fail: Transaction costs (brokerage, STT, slippage), "
            "emotional decisions (revenge trading, FOMO), competing against algos and HFTs. "
            "F&O (Futures & Options): Leverage amplifies both gains and losses. "
            "A ₹10 lakh position in futures can be taken with ₹1L margin — loss can exceed margin. "
            "SEBI 2024 F&O rule changes: Increased lot sizes, weekly expiry restrictions, higher margins — "
            "aimed at reducing retail speculation losses. "
            "Our chatbot focuses on long-term investing, not intraday trading. "
            "If you're drawn to trading, paper-trade first for 6 months before using real money."
        ),
        "tags": ["intraday", "trading", "fo", "risk", "sebi study", "leverage"],
    },

    # ── PERSONAL FINANCE PRINCIPLES ───────────────────────────────────────────

    {
        "id": "emergency_fund",
        "topic": "personal_finance",
        "title": "Emergency Fund — The Foundation of All Investing",
        "content": (
            "An emergency fund is 3–6 months of living expenses kept in a liquid, safe instrument. "
            "This is NON-NEGOTIABLE before you invest in equities. "
            "Why: Without an emergency fund, any job loss or medical emergency forces you to sell "
            "investments at the worst possible time (which is always when markets are down). "
            "Where to keep it: "
            "1. High-interest savings account (SBITO, Kotak 811, FI Money: 6–7%). "
            "2. Liquid mutual fund (better returns, T+1 withdrawal). "
            "3. Flexi FD (for slightly higher returns, instant withdrawal). "
            "NOT in equity mutual funds or stocks — these can fall 30% just when you need cash most. "
            "Size: 3 months if you have stable job + health insurance + dependents few. "
            "6 months if self-employed, single income, or have dependents. "
            "12 months if running a startup or expecting major life changes. "
            "Review annually and top up as your expenses grow."
        ),
        "tags": ["emergency fund", "personal finance", "liquid fund", "financial planning"],
    },
    {
        "id": "insurance_importance",
        "topic": "personal_finance",
        "title": "Insurance — Protecting Your Portfolio",
        "content": (
            "Insurance is not an investment — it's risk transfer. Mixing them is the biggest mistake. "
            "Two types every Indian investor needs: "
            "1. Term Life Insurance: Pure protection, no maturity benefit, very low premium. "
            "   Cover amount: 10–15x annual income. e.g., ₹50L income → ₹5–7.5 crore cover. "
            "   Best providers: LIC Tech Term, Max Life Smart, HDFC Click2Protect. "
            "   Compare on PolicyBazaar. DO NOT buy endowment or ULIPs — terrible returns + high commission. "
            "2. Health Insurance (Mediclaim): Essential — healthcare inflation is 12–15% annually in India. "
            "   Individual cover: ₹10L minimum. Family floater: ₹15–25L. "
            "   Super top-up: cheap way to extend base cover for large medical bills. "
            "Return of Premium (ROP) term plans sound attractive but cost 3–4x more. "
            "The extra premium invested in equity would give far better returns. Take pure term. "
            "Investment-linked insurance (ULIP, endowment) should generally be avoided — "
            "lock-in periods, high charges, and poor transparency compared to direct equity + term insurance."
        ),
        "tags": ["insurance", "term life", "health insurance", "ulip", "personal finance"],
    },
    {
        "id": "inflation_impact",
        "topic": "personal_finance",
        "title": "Inflation — The Silent Wealth Eroder",
        "content": (
            "India's average retail CPI inflation is 5–6% annually. "
            "At 6% inflation: "
            "₹1 lakh today = ₹74,000 purchasing power in 5 years. "
            "₹1 lakh today = ₹31,000 purchasing power in 20 years. "
            "This means your emergency fund in a savings account (3%) is actually LOSING real value. "
            "To preserve wealth: You must earn > inflation after tax. "
            "Post-tax return > inflation: "
            "FD at 7%: After 30% tax slab = 4.9% net. Inflation 6% = LOSING real value. "
            "Equity long-term return (12–15%): After LTCG = ~11–13%. Beats inflation comfortably. "
            "Gold: Roughly tracks inflation over long periods. No real wealth creation. "
            "Real estate: Beats inflation (8–10% CAGR avg in good locations) but illiquid. "
            "Key insight: Cash is NOT safe — it's slowly burning. Investing in equity is the risk of "
            "volatility vs. the certainty of inflation silently destroying cash."
        ),
        "tags": ["inflation", "purchasing power", "real return", "wealth", "personal finance"],
    },

    # ── COMPANY RESEARCH ──────────────────────────────────────────────────────

    {
        "id": "how_to_research_stock",
        "topic": "research",
        "title": "How to Research a Stock Before Investing",
        "content": (
            "Step-by-step stock research process: "
            "1. Understand the business: What does it sell? Who are customers? What is the moat? "
            "   Read annual report 'Letter to Shareholders' and MD&A section. "
            "2. Financials (last 5–10 years): Revenue growth, PAT growth, ROE/ROCE trend, D/E, cash flow. "
            "   Red flag: profit shows growth but cash flow is stagnant (possible accounting manipulation). "
            "3. Valuation: P/E vs history, vs peers, growth (PEG). "
            "4. Management quality: Promoter holding %, promoter pledge %, related-party transactions, "
            "   corporate governance issues. High promoter pledge is a major red flag. "
            "5. Risks: Competition, regulation, cyclicality, technology disruption. "
            "6. Narrative sanity check: Is the 'story' driving the stock price backed by fundamentals? "
            "Where to find data: Screener.in (free, excellent), Tickertape, Moneycontrol, "
            "BSE/NSE investor relations page, SEBI disclosures. "
            "Time needed: Minimum 2–3 hours of research before putting ₹50,000+ in any stock."
        ),
        "tags": ["stock research", "fundamental analysis", "annual report", "screener", "due diligence"],
    },
    {
        "id": "promoter_pledging",
        "topic": "research",
        "title": "Promoter Pledging — A Hidden Risk Signal",
        "content": (
            "Promoter pledging = promoters of a company using their shares as collateral for personal/business loans. "
            "Why it's risky for investors: "
            "If stock price falls, lenders issue margin calls — promoters must pledge more shares or repay. "
            "If they can't, lenders SELL the pledged shares in the open market, crashing the stock further. "
            "This creates a vicious cycle: stock falls → more margin calls → more selling → more falls. "
            "Famous examples: Zee Entertainment, Essel Group, Indiabulls Real Estate — all had "
            "massive pledged holdings that collapsed catastrophically. "
            "What to check: SEBI mandates quarterly disclosure of promoter pledging % on NSE/BSE. "
            "Red flag: Promoter pledge > 20% of their shareholding. "
            "Danger zone: Pledge > 50% or rising trend of pledging. "
            "Best: Zero pledging with promoters actively buying (insider purchases signal confidence)."
        ),
        "tags": ["promoter pledging", "risk", "research", "corporate governance"],
    },

    # ── BROKERS & PLATFORMS ───────────────────────────────────────────────────

    {
        "id": "zerodha_overview",
        "topic": "broker",
        "title": "Zerodha — India's Largest Discount Broker",
        "content": (
            "Zerodha is India's largest stockbroker by active clients (9M+ clients as of 2024). "
            "Founded by Nithin Kamath in 2010. Headquartered in Bengaluru. "
            "Key features: "
            "Zero brokerage on equity delivery trades (buying and holding stocks). "
            "₹20 flat or 0.03% (whichever lower) for intraday and F&O trades. "
            "Kite: Their flagship trading/investing platform (web + mobile). "
            "Coin: Direct mutual fund investment platform (no commission). "
            "Console: Portfolio and analytics dashboard. "
            "Varsity: Free educational platform — one of India's best stock market learning resources. "
            "API: Kite Connect API allows algorithmic trading and building apps like StockMate. "
            "Client funds: Held with HDFC Bank (safe). Demat with CDSL. "
            "Risks: Being a discount broker, you get minimal personal advisory. "
            "For active traders, Zerodha charges stack up (DP charges ₹13.5/sell transaction). "
            "SEBI-registered: NSE, BSE, MCX. AMFI-registered for MF distribution."
        ),
        "tags": ["zerodha", "broker", "kite", "discount broker", "trading platform"],
    },
    {
        "id": "demat_account",
        "topic": "broker",
        "title": "Demat Account — Your Digital Share Safe",
        "content": (
            "A Demat (Dematerialized) account holds your shares and securities in electronic form. "
            "Think of it as a bank account, but for stocks. "
            "Components of investing infrastructure: "
            "1. Bank account: Source of money. "
            "2. Trading account (with broker): Platform to place buy/sell orders. "
            "3. Demat account: Holds your purchased shares. "
            "In India, depositories are CDSL (Central Depository Services Ltd) and NSDL (National Securities Depository Ltd). "
            "Most discount brokers use CDSL. "
            "Annual Maintenance Charge (AMC): ₹300–700/year depending on your broker. "
            "DP (Depository Participant) charge: Charged when you SELL shares from demat — typically ₹13.5–15. "
            "Not charged on buying or for mutual funds held in demat. "
            "One person can have multiple demat accounts with different brokers. "
            "Nomination is now mandatory for demat accounts (SEBI 2023)."
        ),
        "tags": ["demat", "cdsl", "nsdl", "broker", "shares", "account"],
    },

    # ── ZERODHA / KITE SPECIFIC ───────────────────────────────────────────────

    {
        "id": "kite_holdings_explained",
        "topic": "broker",
        "title": "Understanding Holdings in Kite (Zerodha)",
        "content": (
            "In Zerodha Kite, 'Holdings' shows your long-term equity delivery positions (settled in demat). "
            "Key columns: "
            "Instrument: Stock symbol (e.g., GOLDBEES, INFY). "
            "Qty: Number of shares you hold. "
            "Avg. cost: Your weighted average buy price including all buys. "
            "LTP (Last Traded Price): Current market price. "
            "Current value: Qty × LTP. "
            "P&L: (LTP - Avg cost) × Qty = unrealized profit/loss. "
            "Day's change: How much the stock moved today vs yesterday's close. "
            "P&L % = (LTP - Avg cost) / Avg cost × 100. Negative means in loss. "
            "Difference between Holdings and Positions: "
            "Holdings = delivery shares (T+1 settled, in demat, long-term). "
            "Positions = intraday or short-term F&O positions (not in demat). "
            "Console.zerodha.com has detailed P&L reports, tax P&L statement for CA, "
            "and reconciliation of all transactions."
        ),
        "tags": ["zerodha", "kite", "holdings", "portfolio", "demat", "p&l"],
    },

    # ── ECONOMY & MACRO ───────────────────────────────────────────────────────

    {
        "id": "macro_interest_rates",
        "topic": "macro",
        "title": "Interest Rates and Stock Markets",
        "content": (
            "Interest rates (set by RBI's Monetary Policy Committee) are the most powerful macro driver "
            "of equity valuations. "
            "Rate hike cycle effects: "
            "Higher rates → higher risk-free return (FD/G-Sec) → equities become relatively less attractive. "
            "Higher rates → higher borrowing costs → lower corporate profits → lower EPS → lower P/E justified. "
            "High duration debt funds fall sharply when rates rise. "
            "Rate cut cycle effects (bullish for markets): "
            "Lower rates → cheaper borrowing → higher corporate profits. "
            "P/E expansion (same earnings worth more). "
            "Debt funds rally. "
            "RBI rate signals (MPC meetings every 6–8 weeks): CRR, Repo Rate, SDF. "
            "2022–23: RBI raised repo rate from 4% to 6.5% to fight inflation — markets stayed resilient. "
            "2024: Expectations of rate cuts (global and India) supported equity rally. "
            "Sector sensitivity to rates: "
            "Highly sensitive: Banks, NBFCs, Real Estate, Infrastructure. "
            "Less sensitive: FMCG, IT (revenues in USD), Pharma."
        ),
        "tags": ["interest rates", "rbi", "repo rate", "macro", "equity", "economy"],
    },
    {
        "id": "macro_rupee_dollar",
        "topic": "macro",
        "title": "USD/INR and Its Impact on Indian Equities",
        "content": (
            "The exchange rate between USD and INR (USDINR) affects Indian equities in multiple ways. "
            "Rupee depreciation (INR weakens vs USD): "
            "Good for IT companies: TCS, Infosys, Wipro earn in USD — INR revenue rises automatically. "
            "Good for pharma exporters, textile exporters. "
            "Bad for importers: Oil companies (BPCL, IOC), metal companies importing raw materials, "
            "airlines (jet fuel priced in USD). "
            "FIIs feel currency loss: If rupee falls 5%, a foreign investor in Indian stocks loses 5% "
            "in USD terms even if stocks are flat — triggering FII selling. "
            "Rupee appreciation (INR strengthens vs USD): "
            "Reverse of above — bad for exporters, good for importers. "
            "Current account deficit: India is a net importer — chronic CAD puts downward pressure on rupee. "
            "RBI intervenes by selling forex reserves to support rupee during sharp depreciations."
        ),
        "tags": ["currency", "rupee", "dollar", "usdinr", "it sector", "exports", "fii"],
    },

    # ── SECTOR INSIGHTS ───────────────────────────────────────────────────────

    {
        "id": "sector_it_india",
        "topic": "sector",
        "title": "Indian IT Sector — The Export Powerhouse",
        "content": (
            "India's IT services sector (Nifty IT index) is the country's largest export industry and "
            "one of the most globally competitive. "
            "Key companies: TCS (largest by market cap), Infosys, Wipro, HCLTech, LTIMindtree, Mphasis, Persistent. "
            "Revenue model: Largely in USD (FY revenues). "
            "TCS FY24: ~$29 billion revenue. Infosys: ~$19 billion. "
            "Drivers: Global IT spending by US/Europe enterprises, AI/cloud adoption, cost arbitrage. "
            "Risks: US recession (reduces IT budgets), visa restrictions (H-1B), rupee appreciation, "
            "AI automation reducing headcount needs. "
            "Valuation: IT sector trades at premium P/E (25–35x) due to high ROE, zero debt, strong cash flows. "
            "Seasonality: Q1 (Apr-Jun) typically weakest due to furloughs. Q3 (Oct-Dec) strongest. "
            "FY outlook driven by: Revenue guidance (esp. Infosys guides quarterly), demand commentary "
            "on decision-making timelines, AI bookings growth."
        ),
        "tags": ["IT", "tech", "tcs", "infosys", "sector", "exports", "software"],
    },
    {
        "id": "sector_banking",
        "topic": "sector",
        "title": "Indian Banking Sector — Key Metrics to Understand",
        "content": (
            "Banking is the largest weight in Nifty 50 (~33% via financials). Understanding banks is crucial. "
            "Key metrics: "
            "NIM (Net Interest Margin): Interest income - interest expense / avg assets. Higher = better. "
            "Private banks target 3.5–4.5% NIM. PSU banks: 2.5–3.5%. "
            "GNPA / NNPA: Gross/Net Non-Performing Assets %. Lower = better asset quality. "
            "GNPA > 5% = concern. Post-2017 cleanup, major private banks are at 1–2%. "
            "CASA Ratio: %  of deposits in Current Account (0% interest) + Savings Account (3-4%). "
            "Higher CASA = cheaper deposits = better margins. HDFC Bank CASA: ~45%, Yes Bank: Lower. "
            "Capital Adequacy Ratio (CAR / CRAR): Must be >11.5% (SEBI mandate). Buffer for bad loans. "
            "ROA (Return on Assets): Good banks >1.5%. "
            "PCR (Provision Coverage Ratio): How much of bad loans are already provisioned. >70% = healthy."
        ),
        "tags": ["banking", "hdfc bank", "sbi", "npa", "nim", "sector", "finance"],
    },
    {
        "id": "sector_pharma",
        "topic": "sector",
        "title": "Indian Pharma Sector Overview",
        "content": (
            "India is the world's largest generic drug manufacturer — 'pharmacy of the world'. "
            "Key companies: Sun Pharma (largest), Dr. Reddy's, Cipla, Divi's Laboratories, "
            "Aurobindo, Alkem Labs, Torrent Pharma. "
            "Two revenue streams: "
            "1. Domestic branded generics: Higher margins, steady growth, prescriptions from doctors. "
            "2. US generics: High competition, price erosion, FDA compliance-critical. "
            "Key risks: US FDA observations (warning letters, 483s), price erosion in US generics, "
            "China + Eastern Europe competition. "
            "Positives: Rupee depreciation helps (US dollar revenues), patent cliff opportunities "
            "(can manufacture generics after major drug patents expire). "
            "USFDA watch: Any 'Form 483' or 'warning letter' can cause 15–30% stock crash overnight. "
            "Always check company's key manufacturing plants — if US generics are major revenue, "
            "FDA compliance status of those plants is critical due diligence."
        ),
        "tags": ["pharma", "generic drugs", "usfda", "sun pharma", "dr reddy", "sector"],
    },
    {
        "id": "sector_fmcg",
        "topic": "sector",
        "title": "FMCG Sector — Defensive and Premium-Priced",
        "content": (
            "FMCG (Fast Moving Consumer Goods) = daily-use products: soap, shampoo, biscuits, detergent, etc. "
            "Key Indian FMCG companies: HUL (Hindustan Unilever), ITC, Nestle, Britannia, Dabur, Marico, Colgate. "
            "Why FMCG is called 'defensive': Revenue is resilient even during recessions — people still buy soap. "
            "Characteristics: "
            "High brand moats, pricing power, asset-light (don't manufacture everything). "
            "High dividend payers — cash generative. "
            "Low capex requirements. "
            "ROE typically 30–70% for leaders like HUL. "
            "Risks: "
            "Premium valuations (P/E 40–60x) — any earnings miss causes sharp correction. "
            "Rural slowdown (50%+ revenue from rural) — impacted during inflation spikes. "
            "New competition from D2C brands (Mamaearth, Sugar Cosmetics). "
            "Raw material (palm oil, wheat, crude) price volatility. "
            "FMCG competes with market when markets are flat — outperforms in downturns, underperforms in bull runs."
        ),
        "tags": ["fmcg", "hul", "itc", "colgate", "defensive", "consumer goods", "sector"],
    },
]
