import pandas as pd
import numpy as np
from stockstats import StockDataFrame

def prepare_indicators(csvfile):
    '''Prepare the StockDataFrame structure
    StockDataFrame can generate trading metrics on the fly
    ''' 
    df = pd.read_csv(csvfile).astype(float)
    df = df.dropna().reset_index(drop=True)
    sdf = StockDataFrame.retype(df)
    sdf.rename(columns={'volume_(currency)': 'volume'}, inplace=True)
    sdf.rename(columns={'volume_(btc)': 'amount'}, inplace=True)
    sdf.sort_values('timestamp')
    sdf = add_indicators(sdf)
    
    # Drop NaN that appeared in new columns
    sdf.replace([np.inf, -np.inf], np.nan)
    sdf = sdf.dropna().reset_index(drop=True)
    return sdf


def add_indicators(sdf):
    '''sdf is a StockDataFrame object, calling sdf['metric']
    will compute the metric and add it to the StockDataFrame'''

    basic_indicators = [
        # Open price
        sdf.get('open'),
        # High price
        sdf.get('high'),
        # Low price
        sdf.get('low'),
        # Close price
        sdf.get('close'),
        # Volume 
        sdf.get('volume'),
    ]
    extra_indicators = [
        # RSI: Relative Strength Index
        sdf.get('rsi_6'),
        sdf.get('rsi_12'),
        sdf.get('rsi_24'),

        # MACD: Moving Average Convergence Divergence
        # MACD_EMA_SHORT: default to 12
        # MACD_EMA_LONG: default to 26
        # MACD_EMA_SIGNAL: default to 9
        sdf.get('macd'),

        # Bollinger Bands: confident intervals
        # BOLL_WINDOW: default to 20
        # BOLL_STD_TIMES: default to 2
        sdf.get('boll'),
        sdf.get('boll_ub'),
        sdf.get('boll_lb'),

        # KDJ: stochastic oscillator
        # KDJ_WINDOW: default to 9
        sdf.get('kdjk'),
        sdf.get('kdjd'),
        sdf.get('kdjj'),

        # CR
        # CR_MA1: default to 5
        # CR_MA2: default to 10
        # CR_MA3: default to 20
        sdf.get('cr'),
        sdf.get('cr-ma1'),
        sdf.get('cr-ma2'),
        sdf.get('cr-ma3'),
    ]
    return sdf