# Analyzing Finanacial Methods

### Introduction/Background
Today, investment analysis, such as in evaluating stock position strategies, is heavily dependent on technical indicators: functions that extract predefined features from time series. The finance industry relies on technical indicators to quantify time series growth, volatility, patterns, and over/under valued limits.

### Problem definition
Unfortunately, relying on technical indicators does not always result in profitable investments. This is partially due to the noise and natural variability in financial markets, though we aim to determine the effectiveness of technical indicators with regards to extraction of meaningful, informative features. Diving one level deeper, we wish to model which general investment strategies are most suitable for given indicator values, evaluating the contribution of technical indicators to strategy profitability.

### Methods
We will begin by scraping time series data for a large number of stocks. For each timestep[^3] of each stock collected, we will generate a series of technical indicators[^1] which are commonly used to inform a stock trader’s decision-making. This 2-dimensional matrix will act as a time-based feature vector for each stock.
#### Unsupervised Learning: GMM
- We will cluster stocks based on technical indicators (GMM). If the clusters are not well-formed (evaluated using a Silhouette matrix), then technical indicators may not be the best for making trading decisions. Otherwise, there may be n distinct trading policies (strategies), where each cluster corresponds to one policy. Regardless of this outcome, we will continue our project to draw more solid conclusions about the usefulness of technical indicators. We will formulate the trading policies based on the results of the clustering.


#### Supervised Learning: Deep Learning
- We will begin by simulating what our profit/loss would be for every stock if we applied each trading policy over 3-month periods. This will give us a ground truth measure for how well each trading policy performs on each stock. Using the technical indicators computed earlier as our feature set, we will create a deep LSTM neural network[^1] to predict which trading strategy to use for any given stock. The input will be the set of features for a stock over a 3-month period, and the network output will return the optimal trading policy for that stock.

### Potential Results and Discussion
This project will provide us with two results. 
- The first will allow us to understand the effects of technical indicators and if they may be easily used to predict an optimal stock trading policy. Moreover, it will tell us how many policies can be derived effectively from the indicator inputs and help determine which policies provide similar results. We will find this through soft clustering of each datapoint, as this will optimally define the number of clusters without a hyperparameter.
- The second result will also allow us to understand how technical indicators classify into policies directly. This will enable us to evaluate the role indicators play in investment gain. To achieve this, we will be using neural nets that take in these technical indicators as inputs and provide us with patterns and classification of said inputs into each policy. 

### Proposed Timeline
- Weeks 0-1: Data Collection [Dylan]
- Weeks 1-2: Feature Extraction (Technical Indicators) [Blake]
- Weeks 2-3: GMM Clustering [Munim]
- Weeks 3-4: Trading Policy Formulation [Michael]
- Weeks 4-5: Simulation of Ground-Truth Data [Austin]
- Weeks 5-6: Policy Prediction using Deep Neural Networks [Dylan/Austin]
- Weeks 6-7: Results & Analysis [Blake]

### References
[^1]: P. Oncharoen and P. Vateekul, "Deep Learning for Stock Market Prediction Using Event Embedding and Technical Indicators," 2018 5th International Conference on Advanced Informatics: Concept Theory and Applications (ICAICTA), 2018, pp. 19-24, doi: 10.1109/ICAICTA.2018.8541310.

[^2]: Zhai Y., Hsu A., Halgamuge S.K. (2007) Combining News and Technical Indicators in Daily Stock Price Trends Prediction. In: Liu D., Fei S., Hou Z., Zhang H., Sun C. (eds) Advances in Neural Networks – ISNN 2007. ISNN 2007. Lecture Notes in Computer Science, vol 4493. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-72395-0_132

[^3]: Yauheniya Shynkevich, T.M. McGinnity, Sonya A. Coleman, Ammar Belatreche, Yuhua Li, “Forecasting price movements using technical indicators: Investigating the impact of varying input window length, ” Neurocomputing, 2017, pp. 71-88, https://doi.org/10.1016/j.neucom.2016.11.095
