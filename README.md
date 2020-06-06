# Trading bot

## Approach
The goal here is to design a bot that learns trading on Bitcoin using deep reinforcement learning.
An agent following a policy network is designed to choose between 3 actions: Buy, Sell or Hold. The policy network
is represented as an LSTM which is optimized using REINFORCE.

The objective of the agent is to maximize the expected reward r<sub>t</sub> using gradient ascent. The reward r<sub>t</sub> is attributed to
the agent at each time iteration by measuring the variation of its portfolio.
<img src="https://bit.ly/375cTAU" align="center" border="0" alt="J(\theta) = E[\sum_{t=0}^{t=T} r_t]" width="122" height="53" />

The gradient of the objective function is derived using REINFORCE, which is approximated as follows 
<img src="https://bit.ly/3h0Id8F" align="center" border="0" alt="\nabla_\theta J(\theta) = \sum_{t=0}^{t=T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t" width="264" height="53" />

## Network architecture
The state s<sub>t</sub> is defined as being the last 60 feature vectors [x<sub>t-59</sub> ... x<sub>t</sub>].
Each feature vector x<sub>t</sub> contains the following information:
- open  
- high  
- low  
- close  
- volume  
- rsi_6  
- rsi_12  
- rsi_24
- macd  
- boll  
- boll_ub  
- boll_lb  
- kdjk  
- kdjd  
- kdjj  
- portfolio_status  

The state s<sub>t</sub> is the input of the LSTM network. The output is two-fold:
- the first value is a Softmax vector of length 3, telling if the agent should buy, sell or hold its assets
- the second value is a Softmax vector of length 10, telling what portion of the portfolio the agent will buy or sell (according to the first output). This value ranges from 1 to 10 tenth of the portfolio

## Data
The data used is are BTC/USD price with 1 min time step but the program works with any dataset following the same format
Good site for datasets: https://www.cryptodatadownload.com/

The following structure is expected as input in a csv format:

__Columns (Datatype)__:  
- Timestamp (int64)
- Open (float64)
- High (float64)
- Low (float64)
- Close (float64)
- Volume_(BTC) (float64)
- Volume_(Currency) (float64)

## References
For REINFORCE:
https://github.com/cyoon1729/Reinforcement-learning

For Trading environment:
https://github.com/notadamking/RLTrader