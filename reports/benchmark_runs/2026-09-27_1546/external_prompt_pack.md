# External LLM prompt pack (ChatGPT / Claude)

One fresh chat per prompt, web browsing OFF. Record the FINAL line and wall-clock seconds.

## upstox_snapshot:total_invested

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

What is the total invested value of my portfolio (sum of quantity x average price)?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## upstox_snapshot:total_current_value

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

What is the total current value of my portfolio (sum of quantity x last price)?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## upstox_snapshot:total_pnl

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

What is the total profit or loss of my portfolio in rupees?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## upstox_snapshot:total_pnl_pct

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

What is the total percentage profit or loss of my portfolio?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## upstox_snapshot:decision

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

Classify the portfolio as Buy, Hold, Trim, or Exit using this rule: Buy if total P&L% >= 10; Hold if 0 < P&L% < 10; Trim if -10 < P&L% <= 0; Exit if P&L% <= -10.
Show brief working, then end with one final line exactly in the form: FINAL: <Buy|Hold|Trim|Exit>
```

## realistic_8:total_invested

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 42, "average_price": 2456.35, "last_price": 2891.7}, {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 15, "average_price": 3620.1, "last_price": 3412.55}, {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "quantity": 60, "average_price": 1532.8, "last_price": 1678.25}, {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 35, "average_price": 1478.9, "last_price": 1512.4}, {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 250, "average_price": 412.65, "last_price": 438.9}, {"tradingsymbol": "TATAMOTORS", "exchange": "NSE", "quantity": 80, "average_price": 958.4, "last_price": 712.15}, {"tradingsymbol": "SBIN", "exchange": "NSE", "quantity": 120, "average_price": 598.25, "last_price": 811.6}, {"tradingsymbol": "ZOMATO", "exchange": "NSE", "quantity": 400, "average_price": 142.3, "last_price": 238.75}]

What is the total invested value of my portfolio (sum of quantity x average price)?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## realistic_8:total_current_value

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 42, "average_price": 2456.35, "last_price": 2891.7}, {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 15, "average_price": 3620.1, "last_price": 3412.55}, {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "quantity": 60, "average_price": 1532.8, "last_price": 1678.25}, {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 35, "average_price": 1478.9, "last_price": 1512.4}, {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 250, "average_price": 412.65, "last_price": 438.9}, {"tradingsymbol": "TATAMOTORS", "exchange": "NSE", "quantity": 80, "average_price": 958.4, "last_price": 712.15}, {"tradingsymbol": "SBIN", "exchange": "NSE", "quantity": 120, "average_price": 598.25, "last_price": 811.6}, {"tradingsymbol": "ZOMATO", "exchange": "NSE", "quantity": 400, "average_price": 142.3, "last_price": 238.75}]

What is the total current value of my portfolio (sum of quantity x last price)?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## realistic_8:total_pnl

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 42, "average_price": 2456.35, "last_price": 2891.7}, {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 15, "average_price": 3620.1, "last_price": 3412.55}, {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "quantity": 60, "average_price": 1532.8, "last_price": 1678.25}, {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 35, "average_price": 1478.9, "last_price": 1512.4}, {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 250, "average_price": 412.65, "last_price": 438.9}, {"tradingsymbol": "TATAMOTORS", "exchange": "NSE", "quantity": 80, "average_price": 958.4, "last_price": 712.15}, {"tradingsymbol": "SBIN", "exchange": "NSE", "quantity": 120, "average_price": 598.25, "last_price": 811.6}, {"tradingsymbol": "ZOMATO", "exchange": "NSE", "quantity": 400, "average_price": 142.3, "last_price": 238.75}]

What is the total profit or loss of my portfolio in rupees?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## realistic_8:total_pnl_pct

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 42, "average_price": 2456.35, "last_price": 2891.7}, {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 15, "average_price": 3620.1, "last_price": 3412.55}, {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "quantity": 60, "average_price": 1532.8, "last_price": 1678.25}, {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 35, "average_price": 1478.9, "last_price": 1512.4}, {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 250, "average_price": 412.65, "last_price": 438.9}, {"tradingsymbol": "TATAMOTORS", "exchange": "NSE", "quantity": 80, "average_price": 958.4, "last_price": 712.15}, {"tradingsymbol": "SBIN", "exchange": "NSE", "quantity": 120, "average_price": 598.25, "last_price": 811.6}, {"tradingsymbol": "ZOMATO", "exchange": "NSE", "quantity": 400, "average_price": 142.3, "last_price": 238.75}]

What is the total percentage profit or loss of my portfolio?
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## realistic_8:decision

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 42, "average_price": 2456.35, "last_price": 2891.7}, {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 15, "average_price": 3620.1, "last_price": 3412.55}, {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "quantity": 60, "average_price": 1532.8, "last_price": 1678.25}, {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 35, "average_price": 1478.9, "last_price": 1512.4}, {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 250, "average_price": 412.65, "last_price": 438.9}, {"tradingsymbol": "TATAMOTORS", "exchange": "NSE", "quantity": 80, "average_price": 958.4, "last_price": 712.15}, {"tradingsymbol": "SBIN", "exchange": "NSE", "quantity": 120, "average_price": 598.25, "last_price": 811.6}, {"tradingsymbol": "ZOMATO", "exchange": "NSE", "quantity": 400, "average_price": 142.3, "last_price": 238.75}]

Classify the portfolio as Buy, Hold, Trim, or Exit using this rule: Buy if total P&L% >= 10; Hold if 0 < P&L% < 10; Trim if -10 < P&L% <= 0; Exit if P&L% <= -10.
Show brief working, then end with one final line exactly in the form: FINAL: <Buy|Hold|Trim|Exit>
```

## RELIANCE.NS:total_return

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,RELIANCE.NS
2026-06-25,1318.10
2026-06-26,1318.10
2026-06-29,1301.00
2026-06-30,1293.90
2026-07-01,1308.00
2026-07-02,1303.50
2026-07-03,1304.00
2026-07-06,1321.30
2026-07-07,1308.40
2026-07-08,1275.90
2026-07-09,1279.80
2026-07-10,1307.80
2026-07-13,1296.90
2026-07-14,1293.00
2026-07-15,1295.50
2026-07-16,1296.60
2026-07-17,1327.20
2026-07-20,1323.10
2026-07-21,1303.70
2026-07-22,1288.60
2026-07-23,1272.20
2026-07-24,1278.00
2026-07-27,1280.00
2026-07-28,1267.70
2026-07-29,1275.90
2026-07-30,1292.90
2026-07-31,1307.80
2026-08-03,1319.00
2026-08-04,1290.90
2026-08-05,1280.00
2026-08-06,1325.00
2026-08-07,1334.80
2026-08-10,1327.30
2026-08-11,1323.90
2026-08-12,1329.00
2026-08-13,1317.00
2026-08-14,1310.00
2026-08-17,1316.00
2026-08-18,1322.00
2026-08-19,1311.00
2026-08-20,1313.20
2026-08-21,1316.00
2026-08-24,1309.80
2026-08-25,1317.00
2026-08-26,1298.00
2026-08-27,1282.20
2026-08-28,1287.00
2026-08-31,1277.00
2026-09-01,1309.00
2026-09-02,1313.10
2026-09-03,1302.50
2026-09-04,1322.00
2026-09-07,1309.50
2026-09-08,1294.90
2026-09-09,1279.00
2026-09-10,1274.00
2026-09-11,1257.50
2026-09-14,1257.50
2026-09-15,1235.30
2026-09-16,1240.00
2026-09-17,1243.90
2026-09-18,1226.40
2026-09-21,1247.40
2026-09-22,1240.40
2026-09-23,1248.00
2026-09-24,1219.20
2026-09-25,1226.00


What is the 3-month total return (%) of RELIANCE.NS? Use (last close / first close - 1) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## RELIANCE.NS:max_drawdown

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,RELIANCE.NS
2026-06-25,1318.10
2026-06-26,1318.10
2026-06-29,1301.00
2026-06-30,1293.90
2026-07-01,1308.00
2026-07-02,1303.50
2026-07-03,1304.00
2026-07-06,1321.30
2026-07-07,1308.40
2026-07-08,1275.90
2026-07-09,1279.80
2026-07-10,1307.80
2026-07-13,1296.90
2026-07-14,1293.00
2026-07-15,1295.50
2026-07-16,1296.60
2026-07-17,1327.20
2026-07-20,1323.10
2026-07-21,1303.70
2026-07-22,1288.60
2026-07-23,1272.20
2026-07-24,1278.00
2026-07-27,1280.00
2026-07-28,1267.70
2026-07-29,1275.90
2026-07-30,1292.90
2026-07-31,1307.80
2026-08-03,1319.00
2026-08-04,1290.90
2026-08-05,1280.00
2026-08-06,1325.00
2026-08-07,1334.80
2026-08-10,1327.30
2026-08-11,1323.90
2026-08-12,1329.00
2026-08-13,1317.00
2026-08-14,1310.00
2026-08-17,1316.00
2026-08-18,1322.00
2026-08-19,1311.00
2026-08-20,1313.20
2026-08-21,1316.00
2026-08-24,1309.80
2026-08-25,1317.00
2026-08-26,1298.00
2026-08-27,1282.20
2026-08-28,1287.00
2026-08-31,1277.00
2026-09-01,1309.00
2026-09-02,1313.10
2026-09-03,1302.50
2026-09-04,1322.00
2026-09-07,1309.50
2026-09-08,1294.90
2026-09-09,1279.00
2026-09-10,1274.00
2026-09-11,1257.50
2026-09-14,1257.50
2026-09-15,1235.30
2026-09-16,1240.00
2026-09-17,1243.90
2026-09-18,1226.40
2026-09-21,1247.40
2026-09-22,1240.40
2026-09-23,1248.00
2026-09-24,1219.20
2026-09-25,1226.00


What is the maximum drawdown (%) of RELIANCE.NS over the last 3 months? Use the minimum over days of (value / running maximum value - 1) x 100, a negative number.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## RELIANCE.NS:var95

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,RELIANCE.NS
2026-06-25,1318.10
2026-06-26,1318.10
2026-06-29,1301.00
2026-06-30,1293.90
2026-07-01,1308.00
2026-07-02,1303.50
2026-07-03,1304.00
2026-07-06,1321.30
2026-07-07,1308.40
2026-07-08,1275.90
2026-07-09,1279.80
2026-07-10,1307.80
2026-07-13,1296.90
2026-07-14,1293.00
2026-07-15,1295.50
2026-07-16,1296.60
2026-07-17,1327.20
2026-07-20,1323.10
2026-07-21,1303.70
2026-07-22,1288.60
2026-07-23,1272.20
2026-07-24,1278.00
2026-07-27,1280.00
2026-07-28,1267.70
2026-07-29,1275.90
2026-07-30,1292.90
2026-07-31,1307.80
2026-08-03,1319.00
2026-08-04,1290.90
2026-08-05,1280.00
2026-08-06,1325.00
2026-08-07,1334.80
2026-08-10,1327.30
2026-08-11,1323.90
2026-08-12,1329.00
2026-08-13,1317.00
2026-08-14,1310.00
2026-08-17,1316.00
2026-08-18,1322.00
2026-08-19,1311.00
2026-08-20,1313.20
2026-08-21,1316.00
2026-08-24,1309.80
2026-08-25,1317.00
2026-08-26,1298.00
2026-08-27,1282.20
2026-08-28,1287.00
2026-08-31,1277.00
2026-09-01,1309.00
2026-09-02,1313.10
2026-09-03,1302.50
2026-09-04,1322.00
2026-09-07,1309.50
2026-09-08,1294.90
2026-09-09,1279.00
2026-09-10,1274.00
2026-09-11,1257.50
2026-09-14,1257.50
2026-09-15,1235.30
2026-09-16,1240.00
2026-09-17,1243.90
2026-09-18,1226.40
2026-09-21,1247.40
2026-09-22,1240.40
2026-09-23,1248.00
2026-09-24,1219.20
2026-09-25,1226.00


What is the 1-day 95% historical Value at Risk (%) of RELIANCE.NS over the last 3 months? Use daily simple returns; VaR = -(5th percentile of daily returns, linear interpolation) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## RELIANCE.NS:rsi14

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,RELIANCE.NS
2026-06-25,1318.10
2026-06-26,1318.10
2026-06-29,1301.00
2026-06-30,1293.90
2026-07-01,1308.00
2026-07-02,1303.50
2026-07-03,1304.00
2026-07-06,1321.30
2026-07-07,1308.40
2026-07-08,1275.90
2026-07-09,1279.80
2026-07-10,1307.80
2026-07-13,1296.90
2026-07-14,1293.00
2026-07-15,1295.50
2026-07-16,1296.60
2026-07-17,1327.20
2026-07-20,1323.10
2026-07-21,1303.70
2026-07-22,1288.60
2026-07-23,1272.20
2026-07-24,1278.00
2026-07-27,1280.00
2026-07-28,1267.70
2026-07-29,1275.90
2026-07-30,1292.90
2026-07-31,1307.80
2026-08-03,1319.00
2026-08-04,1290.90
2026-08-05,1280.00
2026-08-06,1325.00
2026-08-07,1334.80
2026-08-10,1327.30
2026-08-11,1323.90
2026-08-12,1329.00
2026-08-13,1317.00
2026-08-14,1310.00
2026-08-17,1316.00
2026-08-18,1322.00
2026-08-19,1311.00
2026-08-20,1313.20
2026-08-21,1316.00
2026-08-24,1309.80
2026-08-25,1317.00
2026-08-26,1298.00
2026-08-27,1282.20
2026-08-28,1287.00
2026-08-31,1277.00
2026-09-01,1309.00
2026-09-02,1313.10
2026-09-03,1302.50
2026-09-04,1322.00
2026-09-07,1309.50
2026-09-08,1294.90
2026-09-09,1279.00
2026-09-10,1274.00
2026-09-11,1257.50
2026-09-14,1257.50
2026-09-15,1235.30
2026-09-16,1240.00
2026-09-17,1243.90
2026-09-18,1226.40
2026-09-21,1247.40
2026-09-22,1240.40
2026-09-23,1248.00
2026-09-24,1219.20
2026-09-25,1226.00


What is the 14-day RSI of RELIANCE.NS using the last 3 months of closes? Use simple (non-smoothed) means of the last 14 daily close-to-close gains and losses: RSI = 100 - 100 / (1 + avgGain / avgLoss).
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## TCS.NS:total_return

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,TCS.NS
2026-06-25,2094.70
2026-06-26,2094.70
2026-06-29,2097.90
2026-06-30,2031.50
2026-07-01,1982.60
2026-07-02,2068.10
2026-07-03,2093.50
2026-07-06,2057.60
2026-07-07,2096.10
2026-07-08,2057.50
2026-07-09,2049.50
2026-07-10,2069.00
2026-07-13,2181.50
2026-07-14,2200.60
2026-07-15,2189.20
2026-07-16,2201.00
2026-07-17,2269.00
2026-07-20,2251.10
2026-07-21,2221.10
2026-07-22,2208.30
2026-07-23,2242.90
2026-07-24,2254.30
2026-07-27,2295.60
2026-07-28,2398.00
2026-07-29,2446.60
2026-07-30,2431.80
2026-07-31,2365.60
2026-08-03,2473.70
2026-08-04,2460.00
2026-08-05,2413.00
2026-08-06,2373.00
2026-08-07,2452.70
2026-08-10,2425.70
2026-08-11,2445.70
2026-08-12,2349.70
2026-08-13,2375.00
2026-08-14,2361.00
2026-08-17,2313.20
2026-08-18,2280.00
2026-08-19,2289.00
2026-08-20,2298.00
2026-08-21,2302.00
2026-08-24,2284.10
2026-08-25,2296.20
2026-08-26,2270.00
2026-08-27,2248.40
2026-08-28,2342.00
2026-08-31,2399.30
2026-09-01,2369.00
2026-09-02,2348.00
2026-09-03,2320.10
2026-09-04,2304.00
2026-09-07,2270.00
2026-09-08,2255.50
2026-09-09,2208.00
2026-09-10,2204.10
2026-09-11,2200.80
2026-09-14,2200.80
2026-09-15,2251.00
2026-09-16,2188.80
2026-09-17,2190.00
2026-09-18,2105.00
2026-09-21,2128.70
2026-09-22,2105.00
2026-09-23,2089.60
2026-09-24,2087.00
2026-09-25,2082.00


What is the 3-month total return (%) of TCS.NS? Use (last close / first close - 1) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## TCS.NS:max_drawdown

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,TCS.NS
2026-06-25,2094.70
2026-06-26,2094.70
2026-06-29,2097.90
2026-06-30,2031.50
2026-07-01,1982.60
2026-07-02,2068.10
2026-07-03,2093.50
2026-07-06,2057.60
2026-07-07,2096.10
2026-07-08,2057.50
2026-07-09,2049.50
2026-07-10,2069.00
2026-07-13,2181.50
2026-07-14,2200.60
2026-07-15,2189.20
2026-07-16,2201.00
2026-07-17,2269.00
2026-07-20,2251.10
2026-07-21,2221.10
2026-07-22,2208.30
2026-07-23,2242.90
2026-07-24,2254.30
2026-07-27,2295.60
2026-07-28,2398.00
2026-07-29,2446.60
2026-07-30,2431.80
2026-07-31,2365.60
2026-08-03,2473.70
2026-08-04,2460.00
2026-08-05,2413.00
2026-08-06,2373.00
2026-08-07,2452.70
2026-08-10,2425.70
2026-08-11,2445.70
2026-08-12,2349.70
2026-08-13,2375.00
2026-08-14,2361.00
2026-08-17,2313.20
2026-08-18,2280.00
2026-08-19,2289.00
2026-08-20,2298.00
2026-08-21,2302.00
2026-08-24,2284.10
2026-08-25,2296.20
2026-08-26,2270.00
2026-08-27,2248.40
2026-08-28,2342.00
2026-08-31,2399.30
2026-09-01,2369.00
2026-09-02,2348.00
2026-09-03,2320.10
2026-09-04,2304.00
2026-09-07,2270.00
2026-09-08,2255.50
2026-09-09,2208.00
2026-09-10,2204.10
2026-09-11,2200.80
2026-09-14,2200.80
2026-09-15,2251.00
2026-09-16,2188.80
2026-09-17,2190.00
2026-09-18,2105.00
2026-09-21,2128.70
2026-09-22,2105.00
2026-09-23,2089.60
2026-09-24,2087.00
2026-09-25,2082.00


What is the maximum drawdown (%) of TCS.NS over the last 3 months? Use the minimum over days of (value / running maximum value - 1) x 100, a negative number.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## TCS.NS:var95

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,TCS.NS
2026-06-25,2094.70
2026-06-26,2094.70
2026-06-29,2097.90
2026-06-30,2031.50
2026-07-01,1982.60
2026-07-02,2068.10
2026-07-03,2093.50
2026-07-06,2057.60
2026-07-07,2096.10
2026-07-08,2057.50
2026-07-09,2049.50
2026-07-10,2069.00
2026-07-13,2181.50
2026-07-14,2200.60
2026-07-15,2189.20
2026-07-16,2201.00
2026-07-17,2269.00
2026-07-20,2251.10
2026-07-21,2221.10
2026-07-22,2208.30
2026-07-23,2242.90
2026-07-24,2254.30
2026-07-27,2295.60
2026-07-28,2398.00
2026-07-29,2446.60
2026-07-30,2431.80
2026-07-31,2365.60
2026-08-03,2473.70
2026-08-04,2460.00
2026-08-05,2413.00
2026-08-06,2373.00
2026-08-07,2452.70
2026-08-10,2425.70
2026-08-11,2445.70
2026-08-12,2349.70
2026-08-13,2375.00
2026-08-14,2361.00
2026-08-17,2313.20
2026-08-18,2280.00
2026-08-19,2289.00
2026-08-20,2298.00
2026-08-21,2302.00
2026-08-24,2284.10
2026-08-25,2296.20
2026-08-26,2270.00
2026-08-27,2248.40
2026-08-28,2342.00
2026-08-31,2399.30
2026-09-01,2369.00
2026-09-02,2348.00
2026-09-03,2320.10
2026-09-04,2304.00
2026-09-07,2270.00
2026-09-08,2255.50
2026-09-09,2208.00
2026-09-10,2204.10
2026-09-11,2200.80
2026-09-14,2200.80
2026-09-15,2251.00
2026-09-16,2188.80
2026-09-17,2190.00
2026-09-18,2105.00
2026-09-21,2128.70
2026-09-22,2105.00
2026-09-23,2089.60
2026-09-24,2087.00
2026-09-25,2082.00


What is the 1-day 95% historical Value at Risk (%) of TCS.NS over the last 3 months? Use daily simple returns; VaR = -(5th percentile of daily returns, linear interpolation) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## TCS.NS:rsi14

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,TCS.NS
2026-06-25,2094.70
2026-06-26,2094.70
2026-06-29,2097.90
2026-06-30,2031.50
2026-07-01,1982.60
2026-07-02,2068.10
2026-07-03,2093.50
2026-07-06,2057.60
2026-07-07,2096.10
2026-07-08,2057.50
2026-07-09,2049.50
2026-07-10,2069.00
2026-07-13,2181.50
2026-07-14,2200.60
2026-07-15,2189.20
2026-07-16,2201.00
2026-07-17,2269.00
2026-07-20,2251.10
2026-07-21,2221.10
2026-07-22,2208.30
2026-07-23,2242.90
2026-07-24,2254.30
2026-07-27,2295.60
2026-07-28,2398.00
2026-07-29,2446.60
2026-07-30,2431.80
2026-07-31,2365.60
2026-08-03,2473.70
2026-08-04,2460.00
2026-08-05,2413.00
2026-08-06,2373.00
2026-08-07,2452.70
2026-08-10,2425.70
2026-08-11,2445.70
2026-08-12,2349.70
2026-08-13,2375.00
2026-08-14,2361.00
2026-08-17,2313.20
2026-08-18,2280.00
2026-08-19,2289.00
2026-08-20,2298.00
2026-08-21,2302.00
2026-08-24,2284.10
2026-08-25,2296.20
2026-08-26,2270.00
2026-08-27,2248.40
2026-08-28,2342.00
2026-08-31,2399.30
2026-09-01,2369.00
2026-09-02,2348.00
2026-09-03,2320.10
2026-09-04,2304.00
2026-09-07,2270.00
2026-09-08,2255.50
2026-09-09,2208.00
2026-09-10,2204.10
2026-09-11,2200.80
2026-09-14,2200.80
2026-09-15,2251.00
2026-09-16,2188.80
2026-09-17,2190.00
2026-09-18,2105.00
2026-09-21,2128.70
2026-09-22,2105.00
2026-09-23,2089.60
2026-09-24,2087.00
2026-09-25,2082.00


What is the 14-day RSI of TCS.NS using the last 3 months of closes? Use simple (non-smoothed) means of the last 14 daily close-to-close gains and losses: RSI = 100 - 100 / (1 + avgGain / avgLoss).
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## HDFCBANK.NS:total_return

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,HDFCBANK.NS
2026-06-25,796.30
2026-06-26,796.30
2026-06-29,798.90
2026-06-30,797.95
2026-07-01,796.15
2026-07-02,795.90
2026-07-03,801.05
2026-07-06,829.85
2026-07-07,829.30
2026-07-08,810.30
2026-07-09,817.55
2026-07-10,824.95
2026-07-13,817.95
2026-07-14,809.40
2026-07-15,815.45
2026-07-16,808.30
2026-07-17,819.60
2026-07-20,777.60
2026-07-21,761.45
2026-07-22,753.15
2026-07-23,747.25
2026-07-24,742.80
2026-07-27,739.55
2026-07-28,735.40
2026-07-29,748.20
2026-07-30,753.95
2026-07-31,748.15
2026-08-03,753.00
2026-08-04,742.00
2026-08-05,735.00
2026-08-06,734.30
2026-08-07,731.00
2026-08-10,731.00
2026-08-11,729.00
2026-08-12,729.00
2026-08-13,725.00
2026-08-14,727.00
2026-08-17,729.00
2026-08-18,723.00
2026-08-19,720.00
2026-08-20,725.05
2026-08-21,726.95
2026-08-24,729.00
2026-08-25,727.50
2026-08-26,727.20
2026-08-27,711.00
2026-08-28,720.30
2026-08-31,709.00
2026-09-01,711.90
2026-09-02,700.80
2026-09-03,706.65
2026-09-04,712.10
2026-09-07,710.50
2026-09-08,703.00
2026-09-09,687.10
2026-09-10,693.80
2026-09-11,708.25
2026-09-14,708.25
2026-09-15,716.55
2026-09-16,721.50
2026-09-17,713.00
2026-09-18,731.00
2026-09-21,739.50
2026-09-22,738.60
2026-09-23,737.25
2026-09-24,728.90
2026-09-25,735.60


What is the 3-month total return (%) of HDFCBANK.NS? Use (last close / first close - 1) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## HDFCBANK.NS:max_drawdown

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,HDFCBANK.NS
2026-06-25,796.30
2026-06-26,796.30
2026-06-29,798.90
2026-06-30,797.95
2026-07-01,796.15
2026-07-02,795.90
2026-07-03,801.05
2026-07-06,829.85
2026-07-07,829.30
2026-07-08,810.30
2026-07-09,817.55
2026-07-10,824.95
2026-07-13,817.95
2026-07-14,809.40
2026-07-15,815.45
2026-07-16,808.30
2026-07-17,819.60
2026-07-20,777.60
2026-07-21,761.45
2026-07-22,753.15
2026-07-23,747.25
2026-07-24,742.80
2026-07-27,739.55
2026-07-28,735.40
2026-07-29,748.20
2026-07-30,753.95
2026-07-31,748.15
2026-08-03,753.00
2026-08-04,742.00
2026-08-05,735.00
2026-08-06,734.30
2026-08-07,731.00
2026-08-10,731.00
2026-08-11,729.00
2026-08-12,729.00
2026-08-13,725.00
2026-08-14,727.00
2026-08-17,729.00
2026-08-18,723.00
2026-08-19,720.00
2026-08-20,725.05
2026-08-21,726.95
2026-08-24,729.00
2026-08-25,727.50
2026-08-26,727.20
2026-08-27,711.00
2026-08-28,720.30
2026-08-31,709.00
2026-09-01,711.90
2026-09-02,700.80
2026-09-03,706.65
2026-09-04,712.10
2026-09-07,710.50
2026-09-08,703.00
2026-09-09,687.10
2026-09-10,693.80
2026-09-11,708.25
2026-09-14,708.25
2026-09-15,716.55
2026-09-16,721.50
2026-09-17,713.00
2026-09-18,731.00
2026-09-21,739.50
2026-09-22,738.60
2026-09-23,737.25
2026-09-24,728.90
2026-09-25,735.60


What is the maximum drawdown (%) of HDFCBANK.NS over the last 3 months? Use the minimum over days of (value / running maximum value - 1) x 100, a negative number.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## HDFCBANK.NS:var95

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,HDFCBANK.NS
2026-06-25,796.30
2026-06-26,796.30
2026-06-29,798.90
2026-06-30,797.95
2026-07-01,796.15
2026-07-02,795.90
2026-07-03,801.05
2026-07-06,829.85
2026-07-07,829.30
2026-07-08,810.30
2026-07-09,817.55
2026-07-10,824.95
2026-07-13,817.95
2026-07-14,809.40
2026-07-15,815.45
2026-07-16,808.30
2026-07-17,819.60
2026-07-20,777.60
2026-07-21,761.45
2026-07-22,753.15
2026-07-23,747.25
2026-07-24,742.80
2026-07-27,739.55
2026-07-28,735.40
2026-07-29,748.20
2026-07-30,753.95
2026-07-31,748.15
2026-08-03,753.00
2026-08-04,742.00
2026-08-05,735.00
2026-08-06,734.30
2026-08-07,731.00
2026-08-10,731.00
2026-08-11,729.00
2026-08-12,729.00
2026-08-13,725.00
2026-08-14,727.00
2026-08-17,729.00
2026-08-18,723.00
2026-08-19,720.00
2026-08-20,725.05
2026-08-21,726.95
2026-08-24,729.00
2026-08-25,727.50
2026-08-26,727.20
2026-08-27,711.00
2026-08-28,720.30
2026-08-31,709.00
2026-09-01,711.90
2026-09-02,700.80
2026-09-03,706.65
2026-09-04,712.10
2026-09-07,710.50
2026-09-08,703.00
2026-09-09,687.10
2026-09-10,693.80
2026-09-11,708.25
2026-09-14,708.25
2026-09-15,716.55
2026-09-16,721.50
2026-09-17,713.00
2026-09-18,731.00
2026-09-21,739.50
2026-09-22,738.60
2026-09-23,737.25
2026-09-24,728.90
2026-09-25,735.60


What is the 1-day 95% historical Value at Risk (%) of HDFCBANK.NS over the last 3 months? Use daily simple returns; VaR = -(5th percentile of daily returns, linear interpolation) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## HDFCBANK.NS:rsi14

```
You are a precise financial analyst. Use only the data provided and compute exactly.

Daily closing prices, oldest first (CSV):
Date,HDFCBANK.NS
2026-06-25,796.30
2026-06-26,796.30
2026-06-29,798.90
2026-06-30,797.95
2026-07-01,796.15
2026-07-02,795.90
2026-07-03,801.05
2026-07-06,829.85
2026-07-07,829.30
2026-07-08,810.30
2026-07-09,817.55
2026-07-10,824.95
2026-07-13,817.95
2026-07-14,809.40
2026-07-15,815.45
2026-07-16,808.30
2026-07-17,819.60
2026-07-20,777.60
2026-07-21,761.45
2026-07-22,753.15
2026-07-23,747.25
2026-07-24,742.80
2026-07-27,739.55
2026-07-28,735.40
2026-07-29,748.20
2026-07-30,753.95
2026-07-31,748.15
2026-08-03,753.00
2026-08-04,742.00
2026-08-05,735.00
2026-08-06,734.30
2026-08-07,731.00
2026-08-10,731.00
2026-08-11,729.00
2026-08-12,729.00
2026-08-13,725.00
2026-08-14,727.00
2026-08-17,729.00
2026-08-18,723.00
2026-08-19,720.00
2026-08-20,725.05
2026-08-21,726.95
2026-08-24,729.00
2026-08-25,727.50
2026-08-26,727.20
2026-08-27,711.00
2026-08-28,720.30
2026-08-31,709.00
2026-09-01,711.90
2026-09-02,700.80
2026-09-03,706.65
2026-09-04,712.10
2026-09-07,710.50
2026-09-08,703.00
2026-09-09,687.10
2026-09-10,693.80
2026-09-11,708.25
2026-09-14,708.25
2026-09-15,716.55
2026-09-16,721.50
2026-09-17,713.00
2026-09-18,731.00
2026-09-21,739.50
2026-09-22,738.60
2026-09-23,737.25
2026-09-24,728.90
2026-09-25,735.60


What is the 14-day RSI of HDFCBANK.NS using the last 3 months of closes? Use simple (non-smoothed) means of the last 14 daily close-to-close gains and losses: RSI = 100 - 100 / (1 + avgGain / avgLoss).
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## portfolio:sharpe

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

Daily closing prices, oldest first (CSV):
Date,IDEA.NS,YESBANK.NS,SUZLON.NS
2026-06-25,14.06,24.87,57.14
2026-06-26,14.06,24.87,57.14
2026-06-29,14.43,25.09,57.21
2026-06-30,14.46,24.18,58.90
2026-07-01,14.64,24.58,58.35
2026-07-02,14.48,24.25,57.67
2026-07-03,14.25,24.39,56.87
2026-07-06,14.09,24.26,55.47
2026-07-07,13.85,24.01,54.38
2026-07-08,13.93,23.42,53.13
2026-07-09,14.04,23.64,53.70
2026-07-10,14.19,23.94,53.92
2026-07-13,13.96,23.78,53.17
2026-07-14,13.83,23.61,52.47
2026-07-15,13.67,23.60,52.42
2026-07-16,13.89,23.75,51.97
2026-07-17,13.74,23.61,51.99
2026-07-20,13.51,22.93,53.09
2026-07-21,13.49,23.14,53.17
2026-07-22,13.54,23.24,53.01
2026-07-23,13.20,22.93,52.39
2026-07-24,13.13,22.95,52.28
2026-07-27,13.06,22.94,53.15
2026-07-28,12.95,22.81,48.02
2026-07-29,13.05,22.84,47.45
2026-07-30,12.87,22.54,47.40
2026-07-31,13.01,22.77,48.01
2026-08-03,12.86,23.00,48.10
2026-08-04,12.90,23.00,48.01
2026-08-05,12.80,22.97,48.15
2026-08-06,12.63,22.68,48.10
2026-08-07,12.73,22.69,48.15
2026-08-10,12.92,22.75,47.50
2026-08-11,12.89,22.75,47.50
2026-08-12,13.50,22.98,47.35
2026-08-13,13.51,22.98,47.61
2026-08-14,14.11,22.67,47.10
2026-08-17,13.71,22.92,49.00
2026-08-18,14.12,22.46,47.78
2026-08-19,14.07,22.74,46.77
2026-08-20,14.03,22.74,47.00
2026-08-21,13.94,22.80,46.71
2026-08-24,14.07,22.60,47.05
2026-08-25,15.19,22.50,46.75
2026-08-26,15.06,22.38,46.94
2026-08-27,14.97,22.26,46.81
2026-08-28,14.68,22.10,46.79
2026-08-31,14.33,22.83,47.52
2026-09-01,14.06,22.10,46.50
2026-09-02,14.53,22.15,45.46
2026-09-03,14.48,22.54,46.03
2026-09-04,15.02,22.50,45.35
2026-09-07,15.60,22.42,45.55
2026-09-08,15.42,22.52,45.50
2026-09-09,15.52,22.42,45.45
2026-09-10,14.90,22.30,44.22
2026-09-11,14.97,23.48,44.12
2026-09-14,14.97,23.48,44.12
2026-09-15,14.45,23.07,42.90
2026-09-16,14.26,23.40,42.15
2026-09-17,14.22,23.13,42.54
2026-09-18,13.92,22.72,43.14
2026-09-21,13.84,23.17,43.34
2026-09-22,14.27,23.20,43.00
2026-09-23,14.52,23.24,42.39
2026-09-24,14.12,22.44,40.74
2026-09-25,14.26,22.50,40.80


What is the Sharpe ratio of my portfolio over the last 3 months? Portfolio value each day = sum of (quantity x that day's close) for my current holdings. Use daily simple returns; Sharpe = (mean daily return - 0.05/252) / sample standard deviation of daily returns x sqrt(252).
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## portfolio:var95

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

Daily closing prices, oldest first (CSV):
Date,IDEA.NS,YESBANK.NS,SUZLON.NS
2026-06-25,14.06,24.87,57.14
2026-06-26,14.06,24.87,57.14
2026-06-29,14.43,25.09,57.21
2026-06-30,14.46,24.18,58.90
2026-07-01,14.64,24.58,58.35
2026-07-02,14.48,24.25,57.67
2026-07-03,14.25,24.39,56.87
2026-07-06,14.09,24.26,55.47
2026-07-07,13.85,24.01,54.38
2026-07-08,13.93,23.42,53.13
2026-07-09,14.04,23.64,53.70
2026-07-10,14.19,23.94,53.92
2026-07-13,13.96,23.78,53.17
2026-07-14,13.83,23.61,52.47
2026-07-15,13.67,23.60,52.42
2026-07-16,13.89,23.75,51.97
2026-07-17,13.74,23.61,51.99
2026-07-20,13.51,22.93,53.09
2026-07-21,13.49,23.14,53.17
2026-07-22,13.54,23.24,53.01
2026-07-23,13.20,22.93,52.39
2026-07-24,13.13,22.95,52.28
2026-07-27,13.06,22.94,53.15
2026-07-28,12.95,22.81,48.02
2026-07-29,13.05,22.84,47.45
2026-07-30,12.87,22.54,47.40
2026-07-31,13.01,22.77,48.01
2026-08-03,12.86,23.00,48.10
2026-08-04,12.90,23.00,48.01
2026-08-05,12.80,22.97,48.15
2026-08-06,12.63,22.68,48.10
2026-08-07,12.73,22.69,48.15
2026-08-10,12.92,22.75,47.50
2026-08-11,12.89,22.75,47.50
2026-08-12,13.50,22.98,47.35
2026-08-13,13.51,22.98,47.61
2026-08-14,14.11,22.67,47.10
2026-08-17,13.71,22.92,49.00
2026-08-18,14.12,22.46,47.78
2026-08-19,14.07,22.74,46.77
2026-08-20,14.03,22.74,47.00
2026-08-21,13.94,22.80,46.71
2026-08-24,14.07,22.60,47.05
2026-08-25,15.19,22.50,46.75
2026-08-26,15.06,22.38,46.94
2026-08-27,14.97,22.26,46.81
2026-08-28,14.68,22.10,46.79
2026-08-31,14.33,22.83,47.52
2026-09-01,14.06,22.10,46.50
2026-09-02,14.53,22.15,45.46
2026-09-03,14.48,22.54,46.03
2026-09-04,15.02,22.50,45.35
2026-09-07,15.60,22.42,45.55
2026-09-08,15.42,22.52,45.50
2026-09-09,15.52,22.42,45.45
2026-09-10,14.90,22.30,44.22
2026-09-11,14.97,23.48,44.12
2026-09-14,14.97,23.48,44.12
2026-09-15,14.45,23.07,42.90
2026-09-16,14.26,23.40,42.15
2026-09-17,14.22,23.13,42.54
2026-09-18,13.92,22.72,43.14
2026-09-21,13.84,23.17,43.34
2026-09-22,14.27,23.20,43.00
2026-09-23,14.52,23.24,42.39
2026-09-24,14.12,22.44,40.74
2026-09-25,14.26,22.50,40.80


What is the 1-day 95% historical Value at Risk (%) of my portfolio over the last 3 months? Portfolio value each day = sum of (quantity x that day's close) for my current holdings. Use daily simple returns; VaR = -(5th percentile of daily returns, linear interpolation) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## portfolio:max_drawdown

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

Daily closing prices, oldest first (CSV):
Date,IDEA.NS,YESBANK.NS,SUZLON.NS
2026-06-25,14.06,24.87,57.14
2026-06-26,14.06,24.87,57.14
2026-06-29,14.43,25.09,57.21
2026-06-30,14.46,24.18,58.90
2026-07-01,14.64,24.58,58.35
2026-07-02,14.48,24.25,57.67
2026-07-03,14.25,24.39,56.87
2026-07-06,14.09,24.26,55.47
2026-07-07,13.85,24.01,54.38
2026-07-08,13.93,23.42,53.13
2026-07-09,14.04,23.64,53.70
2026-07-10,14.19,23.94,53.92
2026-07-13,13.96,23.78,53.17
2026-07-14,13.83,23.61,52.47
2026-07-15,13.67,23.60,52.42
2026-07-16,13.89,23.75,51.97
2026-07-17,13.74,23.61,51.99
2026-07-20,13.51,22.93,53.09
2026-07-21,13.49,23.14,53.17
2026-07-22,13.54,23.24,53.01
2026-07-23,13.20,22.93,52.39
2026-07-24,13.13,22.95,52.28
2026-07-27,13.06,22.94,53.15
2026-07-28,12.95,22.81,48.02
2026-07-29,13.05,22.84,47.45
2026-07-30,12.87,22.54,47.40
2026-07-31,13.01,22.77,48.01
2026-08-03,12.86,23.00,48.10
2026-08-04,12.90,23.00,48.01
2026-08-05,12.80,22.97,48.15
2026-08-06,12.63,22.68,48.10
2026-08-07,12.73,22.69,48.15
2026-08-10,12.92,22.75,47.50
2026-08-11,12.89,22.75,47.50
2026-08-12,13.50,22.98,47.35
2026-08-13,13.51,22.98,47.61
2026-08-14,14.11,22.67,47.10
2026-08-17,13.71,22.92,49.00
2026-08-18,14.12,22.46,47.78
2026-08-19,14.07,22.74,46.77
2026-08-20,14.03,22.74,47.00
2026-08-21,13.94,22.80,46.71
2026-08-24,14.07,22.60,47.05
2026-08-25,15.19,22.50,46.75
2026-08-26,15.06,22.38,46.94
2026-08-27,14.97,22.26,46.81
2026-08-28,14.68,22.10,46.79
2026-08-31,14.33,22.83,47.52
2026-09-01,14.06,22.10,46.50
2026-09-02,14.53,22.15,45.46
2026-09-03,14.48,22.54,46.03
2026-09-04,15.02,22.50,45.35
2026-09-07,15.60,22.42,45.55
2026-09-08,15.42,22.52,45.50
2026-09-09,15.52,22.42,45.45
2026-09-10,14.90,22.30,44.22
2026-09-11,14.97,23.48,44.12
2026-09-14,14.97,23.48,44.12
2026-09-15,14.45,23.07,42.90
2026-09-16,14.26,23.40,42.15
2026-09-17,14.22,23.13,42.54
2026-09-18,13.92,22.72,43.14
2026-09-21,13.84,23.17,43.34
2026-09-22,14.27,23.20,43.00
2026-09-23,14.52,23.24,42.39
2026-09-24,14.12,22.44,40.74
2026-09-25,14.26,22.50,40.80


What is the maximum drawdown (%) of my portfolio over the last 3 months? Portfolio value each day = sum of (quantity x that day's close) for my current holdings. Use the minimum over days of (value / running maximum value - 1) x 100, a negative number.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

## portfolio:total_return

```
You are a precise financial analyst. Use only the data provided and compute exactly.

My portfolio holdings (JSON, Upstox schema):
[{"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15}, {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23}, {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45}]

Daily closing prices, oldest first (CSV):
Date,IDEA.NS,YESBANK.NS,SUZLON.NS
2026-06-25,14.06,24.87,57.14
2026-06-26,14.06,24.87,57.14
2026-06-29,14.43,25.09,57.21
2026-06-30,14.46,24.18,58.90
2026-07-01,14.64,24.58,58.35
2026-07-02,14.48,24.25,57.67
2026-07-03,14.25,24.39,56.87
2026-07-06,14.09,24.26,55.47
2026-07-07,13.85,24.01,54.38
2026-07-08,13.93,23.42,53.13
2026-07-09,14.04,23.64,53.70
2026-07-10,14.19,23.94,53.92
2026-07-13,13.96,23.78,53.17
2026-07-14,13.83,23.61,52.47
2026-07-15,13.67,23.60,52.42
2026-07-16,13.89,23.75,51.97
2026-07-17,13.74,23.61,51.99
2026-07-20,13.51,22.93,53.09
2026-07-21,13.49,23.14,53.17
2026-07-22,13.54,23.24,53.01
2026-07-23,13.20,22.93,52.39
2026-07-24,13.13,22.95,52.28
2026-07-27,13.06,22.94,53.15
2026-07-28,12.95,22.81,48.02
2026-07-29,13.05,22.84,47.45
2026-07-30,12.87,22.54,47.40
2026-07-31,13.01,22.77,48.01
2026-08-03,12.86,23.00,48.10
2026-08-04,12.90,23.00,48.01
2026-08-05,12.80,22.97,48.15
2026-08-06,12.63,22.68,48.10
2026-08-07,12.73,22.69,48.15
2026-08-10,12.92,22.75,47.50
2026-08-11,12.89,22.75,47.50
2026-08-12,13.50,22.98,47.35
2026-08-13,13.51,22.98,47.61
2026-08-14,14.11,22.67,47.10
2026-08-17,13.71,22.92,49.00
2026-08-18,14.12,22.46,47.78
2026-08-19,14.07,22.74,46.77
2026-08-20,14.03,22.74,47.00
2026-08-21,13.94,22.80,46.71
2026-08-24,14.07,22.60,47.05
2026-08-25,15.19,22.50,46.75
2026-08-26,15.06,22.38,46.94
2026-08-27,14.97,22.26,46.81
2026-08-28,14.68,22.10,46.79
2026-08-31,14.33,22.83,47.52
2026-09-01,14.06,22.10,46.50
2026-09-02,14.53,22.15,45.46
2026-09-03,14.48,22.54,46.03
2026-09-04,15.02,22.50,45.35
2026-09-07,15.60,22.42,45.55
2026-09-08,15.42,22.52,45.50
2026-09-09,15.52,22.42,45.45
2026-09-10,14.90,22.30,44.22
2026-09-11,14.97,23.48,44.12
2026-09-14,14.97,23.48,44.12
2026-09-15,14.45,23.07,42.90
2026-09-16,14.26,23.40,42.15
2026-09-17,14.22,23.13,42.54
2026-09-18,13.92,22.72,43.14
2026-09-21,13.84,23.17,43.34
2026-09-22,14.27,23.20,43.00
2026-09-23,14.52,23.24,42.39
2026-09-24,14.12,22.44,40.74
2026-09-25,14.26,22.50,40.80


What is the 3-month total return (%) of my portfolio? Portfolio value each day = sum of (quantity x that day's close) for my current holdings. Use (last close / first close - 1) x 100.
Show brief working, then end with one final line exactly in the form: FINAL: <number>
```

