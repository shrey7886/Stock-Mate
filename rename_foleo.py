import os

files = [
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\src\pages\LoginPage.jsx",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\src\pages\LandingPage.jsx",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\src\pages\DashboardPage.jsx",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\src\pages\ChatPage.jsx",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\src\pages\BrokerPage.jsx",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\src\components\AppShell.jsx",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\frontend\index.html",
    r"c:\ML Projects\STOCKMATE\Stock-Mate\backend_api\utils\email.py",
]

for f in files:
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
        content = content.replace("StockMate", "Foleo").replace("Stock-Mate", "Foleo")
        with open(f, "w", encoding="utf-8") as file:
            file.write(content)
