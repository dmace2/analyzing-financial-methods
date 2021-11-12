# Analyzing Finanacial Methods

### Introduction/Background
Today, investment analysis, such as in evaluating stock position strategies, is heavily dependent on technical indicators: functions that extract predefined features from time series. The finance industry relies on technical indicators to quantify time series growth, volatility, patterns, and over/under valued limits.

### Problem Definition
Unfortunately, relying on technical indicators does not always result in profitable investments. This is partially due to the noise and natural variability in financial markets, though we aim to determine the effectiveness of technical indicators with regards to extraction of meaningful, informative features. Diving one level deeper, we wish to model which general investment strategies are most suitable for given indicator values, evaluating the contribution of technical indicators to strategy profitability.

### Data Collection

##### Collecting Stock Data
We began by scraping time series data for a set of 20 stocks in the Consumer Goods industry: Amazon, Tesla, Alibaba, Home Depot, Toyota Motors, NIke, McDonald’s, Lowes, Starbucks, JD.com, Pinduoduo, Booking Holdings, General Motors, Mercadolibre, TJ Max, Nio, Ford Motors, Lululemon Athletica, Honda Motor Co, and Chipotle Mexican Grill. The yFinance Python library allowed us to obtain 10 years of time series data per stock, organized into daily increments. Each data point gives us five values: 
- Opening Price
- Closing Price
- High
- Low
- Volume (Total Shares Traded)

##### Computing Technical Indicators
For each timestep of data collected, we used TA-Lib to collect 14 technical indicators which are commonly used to influence a stock trader’s decision-making strategies. These technical indicators include:
- **Relative Strength Index (RSI):** Oscillates between 0 and 100. Buy signal below 20, Sell signal above 80.
- **Ultimate Oscillator:** Oscillates between 0 and 100. Buy signal below 30, Sell above 70
- **Bollinger Bands:** Width of expected price range, two standard deviations
- **Chaikin Oscillator:** Oscillates between 0 and 100, signals oversold / underbought dependent on Volume
- **Normalized Average True Range (NATR):**  Measure of time series volatility and directional uncertainty
- **Simple Moving Average, 5 day, 20 day, 100 day (SMA)**: Average of last n days
- **Parabolic SAR:** Overlap indicator, buy signal when value jumps from above price to below price, and vice versa
- **Williams %R:** Oscillates between 0 and 100, signifies strength in momentum
- **Absolute Price Oscillator (APO):** Oscillator centered at 0, buy signal when crossing from negative to positive, sell signal when crossing from positive to negative
- **Rate of Change, 5 day, 20 day, 100 day (ROC):** Simple % change in price across a given period of n days

We appended these indicators as new columns of our pandas dataframe, to generate a 10 year by 19 column matrix, where each row corresponded to one day.

### Methods
We will begin by scraping time series data for a large number of stocks. For each timestep[^1] of each stock collected, we will generate a series of technical indicators[^2] which are commonly used to inform a stock trader’s decision-making. This 2-dimensional matrix will act as a time-based feature vector for each stock.
#### Unsupervised Learning: GMM
- We will cluster stocks based on technical indicators (GMM). If the clusters are not well-formed (evaluated using a Silhouette matrix), then technical indicators may not be the best for making trading decisions. Otherwise, there may be n distinct trading policies (strategies), where each cluster corresponds to one policy. Regardless of this outcome, we will continue ouxr project to draw more solid conclusions about the usefulness of technical indicators. We will formulate the trading policies based on the results of the clustering.


#### Supervised Learning: Deep Learning
- We will begin by simulating what our profit/loss would be for every stock if we applied each trading policy over 3-month periods. This will give us a ground truth measure for how well each trading policy performs on each stock. Using the technical indicators computed earlier as our feature set, we will create a deep LSTM neural network[^3] to predict which trading strategy to use for any given stock. The input will be the set of features for a stock over a 3-month period, and the network output will return the optimal trading policy for that stock.

### Potential Results and Discussion
This project will provide us with two results. 
- The first will allow us to understand the effects of technical indicators and if they may be easily used to predict an optimal stock trading policy. Moreover, it will tell us how many policies can be derived effectively from the indicator inputs and help determine which policies provide similar results. We will find this through soft clustering of each datapoint, as this will optimally define the number of clusters without a hyperparameter.
- The second result will also allow us to understand how technical indicators classify into policies directly. This will enable us to evaluate the role indicators play in investment gain. To achieve this, we will be using neural nets that take in these technical indicators as inputs and provide us with patterns and classification of said inputs into each policy. 

### Proposed Timeline
- Weeks 0-1: Data Collection **[Dylan]**
- Weeks 1-2: Feature Extraction (Technical Indicators) **[Blake]**
- Weeks 2-3: GMM Clustering **[Munim]**
- Weeks 3-4: Trading Policy Formulation **[Michael]**
- Weeks 4-5: Simulation of Ground-Truth Data **[Austin]**
- Weeks 5-6: Policy Prediction using Deep Neural Networks **[Dylan/Austin]**
- Weeks 6-7: Results & Analysis **[Blake]**

### References
[^1]: Zhai Y., Hsu A., Halgamuge S.K. (2007) Combining News and Technical Indicators in Daily Stock Price Trends Prediction. In: Liu D., Fei S., Hou Z., Zhang H., Sun C. (eds) Advances in Neural Networks – ISNN 2007. ISNN 2007. Lecture Notes in Computer Science, vol 4493. Springer, Berlin, Heidelberg. 
[^2]: Yauheniya Shynkevich, T.M. McGinnity, Sonya A. Coleman, Ammar Belatreche, Yuhua Li, “Forecasting price movements using technical indicators: Investigating the impact of varying input window length, ” Neurocomputing, 2017, pp. 71-88, https://doi.org/10.1016/j.neucom.2016.11.095.
[^3]: P. Oncharoen and P. Vateekul, "Deep Learning for Stock Market Prediction Using Event Embedding and Technical Indicators," 2018 5th International Conference on Advanced Informatics: Concept Theory and Applications (ICAICTA), 2018, pp. 19-24, doi: 10.1109/ICAICTA.2018.8541310.
