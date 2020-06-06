import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing

MAX_TRADING_SESSION = 5_000  # 1 month


class BitcoinTradingEnv(gym.Env):
    """A Bitcoin trading environment for OpenAI gym
    see original: https://github.com/notadamking/RLTrader
    """
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None
    def __init__(self, sdf, lookback_window_size=60, 
                            commission=1e-4,  
                            initial_balance=1000,
                            serial=False):
        super(BitcoinTradingEnv, self).__init__()
        self.sdf = sdf.dropna().reset_index(drop=True)

        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial

    # Actions of the format Buy 1/10, Sell 3/10, Hold, etc.
        self.action_space = spaces.MultiDiscrete([3, 10])
    # Observes the OHCLV values, net worth, and trade history
        vector_size = 15 + 5 # 5 is length of account history
        self.observation_space = spaces.Box(low=0, high=1, 
                shape=(vector_size, lookback_window_size + 1), dtype=np.float16)

    def _scale_feature(self, feature):
        feature_col = feature.reshape(-1, 1)
        if np.max(feature_col) == 0 and np.min(feature_col) == 0:
            scaled_feature_col = feature_col
        else:
            scaled_feature_col = self.scaler.fit_transform(feature_col)
        return scaled_feature_col.T

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1

        obs = np.array([
            self._scale_feature(self.active_sdf['open'].values[self.current_step:end]),  
            self._scale_feature(self.active_sdf['high'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['low'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['close'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['volume'].values[self.current_step:end]),

            self._scale_feature(self.active_sdf['rsi_6'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['rsi_12'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['rsi_24'].values[self.current_step:end]),

            self._scale_feature(self.active_sdf['macd'].values[self.current_step:end]),

            self._scale_feature(self.active_sdf['boll'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['boll_ub'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['boll_lb'].values[self.current_step:end]),

            self._scale_feature(self.active_sdf['kdjk'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['kdjd'].values[self.current_step:end]),
            self._scale_feature(self.active_sdf['kdjj'].values[self.current_step:end]),

            # self._scale_feature(self.active_sdf['cr'].values[self.current_step:end]),
            # self._scale_feature(self.active_sdf['cr-ma1'].values[self.current_step:end]),
            # self._scale_feature(self.active_sdf['cr-ma2'].values[self.current_step:end]),
            # self._scale_feature(self.active_sdf['cr-ma3'].values[self.current_step:end]),
        ]).squeeze()

        scaled_history = self.scaler.fit_transform(self.account_history.T)
        obs = np.append(obs, scaled_history.T[:, -(self.lookback_window_size
                                                            + 1):], axis=0)
        return obs


    def _reset_session(self):
        self.current_step = 0
        if self.serial:
            self.steps_left = len(self.sdf) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                self.lookback_window_size, len(self.sdf) - self.steps_left)
        self.active_sdf = self.sdf[self.frame_start -   
            self.lookback_window_size:self.frame_start + self.steps_left]


    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10
        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0
        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type < 2:
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price  * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
            'step': self.frame_start + self.current_step,
            'amount': btc_sold if btc_sold > 0 else btc_bought,
            'total': sales if btc_sold > 0 else cost,
            'type': "sell" if btc_sold > 0 else "buy"
            })

        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)


    def _get_price(self, step):
        price = self.active_sdf['close'].values[step + self.lookback_window_size]
        return price


    def _get_current_price(self):
        # current price is the Close price of last input
        current_price = self.active_sdf['close'].values[self.current_step + self.lookback_window_size]
        return current_price


    def _get_market_trend(self):
        current_price = self._get_current_price()
        inital_price = self._get_price(0)
        trend = (current_price - inital_price) / inital_price
        return trend

    
    def _get_reward(self):
        profit = self.net_worth - self.initial_balance
        hold_profit = self._get_market_trend() * self.initial_balance
        reward = profit - hold_profit
        return reward


    def _get_profit(self):
        return self.net_worth - self.initial_balance

    
    def _get_hold_profit(self):
        return self._get_market_trend() * self.initial_balance


    def step(self, action):
        current_price = self._get_current_price()
        self._take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1
        done = self.net_worth <= 0
        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            self._reset_session()
            done = 1
        obs = self._next_observation()
        reward = self._get_reward()
        return obs, reward, done, {}


    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self._reset_session()
        
        self.account_history = np.repeat([
            [self.net_worth],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size + 1, axis=1)
        self.trades = []
        return self._next_observation()

    def render(self):
        # Render the environment to the screen
        profit = self._get_profit()
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Asset: {self.btc_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')
        print(f'Profit HOLD strategy: {self._get_hold_profit()}')
