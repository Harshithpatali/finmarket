from config.universe import STARTER_UNIVERSE, MARKET_TICKER, START_DATE, END_DATE
from src.data.market import download_universe
if __name__ == "__main__":
    download_universe(STARTER_UNIVERSE + [MARKET_TICKER], START_DATE, END_DATE)
