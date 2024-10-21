import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pytickersymbols import PyTickerSymbols

def get_spx_tickers():
    stock_data = PyTickerSymbols()
    tickers = stock_data.get_sp_500_nyc_yahoo_tickers()
    return [ticker.replace('$', '') for ticker in tickers]

def get_market_open_close(date):
    ny_tz = pytz.timezone('America/New_York')
    open_time = ny_tz.localize(datetime.combine(date, datetime.min.time())).replace(hour=9, minute=30)
    close_time = open_time.replace(hour=12, minute=30)  # 3 hours after market open
    return open_time, close_time

def get_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', 'Unknown')
    except Exception as e:
        print(f"Error fetching sector for {ticker}: {e}")
        return 'Unknown'

def calculate_indicators(tickers, date):
    ny_tz = pytz.timezone('America/New_York')
    start_time, end_time = get_market_open_close(date)
    all_data = {}
    sectors = {}

    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_time, end=end_time, interval="1m")
            if not data.empty:
                all_data[ticker] = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                sectors[ticker] = get_sector(ticker)
            else:
                print(f"No data available for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if not all_data:
        print("No data available for any ticker.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Create a common index for all data
    common_index = pd.date_range(start=start_time, end=end_time, freq='1min')

    # Reindex all dataframes to the common index
    for ticker in all_data:
        all_data[ticker] = all_data[ticker].reindex(common_index)

    # Combine all OHLCV data
    combined_data = pd.concat(all_data, axis=1)
    combined_data.columns = pd.MultiIndex.from_product([all_data.keys(), ['Open', 'High', 'Low', 'Close', 'Volume']])

    # Calculate indicators for all stocks
    indicators = calculate_market_indicators(combined_data)

    # Calculate indicators for each sector
    sector_indicators = {}
    for sector in set(sectors.values()):
        sector_tickers = [ticker for ticker, sec in sectors.items() if sec == sector]
        sector_data = combined_data[sector_tickers]
        sector_indicators[sector] = calculate_market_indicators(sector_data)

    sector_indicators_df = pd.concat(sector_indicators, axis=1)
    sector_indicators_df.columns = pd.MultiIndex.from_product([sector_indicators.keys(), sector_indicators[list(sector_indicators.keys())[0]].columns])

    return combined_data, indicators, sector_indicators_df

def calculate_market_indicators(data):
    indicators = pd.DataFrame(index=data.index)
    price_changes = data.xs('Close', axis=1, level=1).pct_change()
    volume_data = data.xs('Volume', axis=1, level=1)

    indicators['Advances'] = (price_changes > 0).sum(axis=1)
    indicators['Declines'] = (price_changes < 0).sum(axis=1)
    indicators['Unchanged'] = (price_changes == 0).sum(axis=1)
    indicators['AD_Line'] = (indicators['Advances'] - indicators['Declines']).cumsum()

    # Calculate cumulative UVOL and DVOL
    uvol_per_period = volume_data[price_changes > 0].sum(axis=1)
    dvol_per_period = volume_data[price_changes < 0].sum(axis=1)
    indicators['UVOL'] = uvol_per_period.cumsum()
    indicators['DVOL'] = dvol_per_period.cumsum()
    indicators['VOLD'] = indicators['UVOL'] - indicators['DVOL']
    indicators['VOLD_Ratio'] = indicators['UVOL'] / indicators['DVOL']

    return indicators

def main():
    tickers = get_spx_tickers()
    print(f"Number of tickers: {len(tickers)}")

    ny_tz = pytz.timezone('America/New_York')
    today = datetime.now(ny_tz).date()

    ohlcv_data, market_indicators, sector_indicators = calculate_indicators(tickers, today)

    date_str = today.strftime('%Y-%m-%d')
    ohlcv_data.to_csv(f'spx_ohlcv_{date_str}.csv')
    market_indicators.to_csv(f'spx_market_indicators_{date_str}.csv')
    sector_indicators.to_csv(f'spx_sector_indicators_{date_str}.csv')

    print(f"Data saved for date: {date_str}")
    print(f"OHLCV data shape: {ohlcv_data.shape}")
    print(f"Market indicators data shape: {market_indicators.shape}")
    print(f"Sector indicators data shape: {sector_indicators.shape}")

if __name__ == "__main__":
    main()
