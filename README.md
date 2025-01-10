# NEPSE Index Forecasting Using Hybrid Deep Learning Models
The codes in this repository use the Keras hyperparameters optimization framework and Hyperband algorithm to automatically tune the hyperparameter of hybrid deep learning models—LSTM-Dense and GRU-Dense models—to forecast the closing NEPSE index using only the past closing index. It uses the dropout and early_stopping regularization techniques to prevent the model from overfitting.

## Introduction
The stock exchange, a critical platform for buying and selling shares, is integral to a nation's economic and financial development. It enables investments, allowing individuals to own company shares, while the broader capital market fosters economic growth by mobilizing long-term funds. Despite its importance, the stock market's inherent volatility, influenced by diverse factors such as economic policies, political events, and global dynamics, presents significant challenges in predicting share price movements. These challenges form the basis for ongoing debates about market efficiency and randomness, central to the Efficient Market Hypothesis (EMH) and Random Walk Hypothesis (RWH). 

The EMH asserts that markets are efficient when prices reflect all available information, rendering price changes random and independent. However, empirical studies reveal inconsistencies in market efficiency across contexts and indicate that stock returns may deviate from random patterns. Additionally, anomalies like calendar effects challenge the EMH by suggesting predictable patterns in stock prices based on specific times, days, or seasons. These discrepancies highlight the need for advanced models capable of addressing the complexities of financial markets, particularly in emerging markets like Nepal, where studies on market efficiency and seasonality are scarce.

Traditional stock price prediction methods, such as the Autoregressive Integrated Moving Average (ARIMA), primarily address linear patterns and require stationary data. However, these approaches are inadequate for handling noisy and nonlinear financial data. In contrast, machine learning (ML) and deep learning (DL) techniques provide robust solutions by identifying intricate patterns in large datasets. When paired with Dense layers in hybrid architectures, techniques like Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) enhance predictive accuracy. Despite these advancements, limited projects have focused on applying and comparing such models to predict the Nepal Stock Exchange (NEPSE) index, prompting this project.

## Project Objectives
This study aims to investigate three critical questions:
Does the NEPSE index exhibit weak-form efficiency, or does its behavior align with the randomness described by the RWH?
Which hybrid model—LSTM-Dense or GRU-Dense—offers superior predictive accuracy for the NEPSE index?
How do single-layer and multi-layer Recurrent Neural Network (RNN) architectures perform in hybrid models, and how can hyperparameter optimization enhance their performance?
The project employs state-of-the-art hybrid DL models to evaluate the weak-form efficiency of the Nepalese stock market. It also uses advanced hyperparameter tuning methods, such as the Hyperband algorithm, to optimize hybrid architectures and provide actionable insights for stock price prediction.

## Project Framework and Methodology
The framework for this project involves:

**Data Preparation:** Historical stock price data is cleaned, scaled using MinMaxScaler(), and split into training, validation, and testing datasets in a 70:10:20 ratio.

**Model Construction:** Hybrid models combining LSTM or GRU with dense layers are developed. The Keras Tuner and Hyperband algorithm optimize hyperparameters, ensuring robust configurations.

**Evaluation:** Model performance is assessed using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared (R²). Time series cross-validation with TimeSeriesSplit() validates model robustness.

**Visualization and Statistical Testing:** Predictions are visualized using Matplotlib. Statistical tests, including Welch’s t-test and D’Agostino-Pearson normality tests, evaluate significant differences between model performances.

## Technologies and Tools
The project leverages Python-based tools, including TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, Matplotlib, SciPy, and Playwright, to implement and evaluate the hybrid models effectively.

## Project Summary and Implications
The study provides critical insights into the NEPSE index's behavior:

**Market Efficiency:** The Nepalese stock market exhibits inefficiency, as historical prices significantly predict future movements, contradicting the RWH.

**Model Performance:** The GRU-Dense model outperforms the LSTM-Dense model in predictive accuracy. Single-layer RNN architectures are effective for hybrid models, capturing nonlinear stock movements efficiently.

**Optimization:** Hyperparameter optimization via Hyperband improves model accuracy, offering a resource-efficient alternative to traditional tuning methods.
Implications for Stakeholders

**Investors:** Historical price data can inform active strategies and technical analysis, enabling more profitable decisions than passive "buy-and-hold" approaches.

**Portfolio Managers:** Active portfolio management becomes critical for identifying undervalued or overvalued stocks, emphasizing rigorous risk assessment and diligence.

**Policymakers:** Addressing market inefficiencies is vital to preventing resource misallocation, mitigating income inequality, and fostering economic growth. Regulatory measures should ensure that market performance aligns with economic realities.

**Technocrats:** Advanced ML and DL techniques can develop sophisticated trading algorithms, improving market predictions and efficiency.

## Contribution to Literature and Practice
This project bridges gaps in financial analytics by evaluating the Nepalese stock market's weak-form efficiency using hybrid DL models. It provides a comparative analysis of LSTM-Dense and GRU-Dense models, emphasizing their applicability to emerging markets. The project also highlights the significance of resource-efficient hyperparameter tuning, offering a novel approach to optimizing predictive models.

## Broader Significance
The findings extend beyond financial applications, offering insights into resource allocation and wealth distribution. Efficient markets channel resources into productive ventures, penalize inefficiencies and promote economic stability. In contrast, inefficient markets exacerbate socioeconomic disparities, emphasizing the need for informed regulatory interventions.

## Conclusion
This project underscores the interplay between market efficiency, predictive modeling, and economic growth. By leveraging advanced DL techniques and optimization methods, it seeks to unravel the complexities of Nepal’s stock market. The findings offer actionable insights for investors, regulators, and policymakers, ensuring informed financial decisions and promoting equitable economic development. Ultimately, efficient markets contribute to stability and progress, aligning with the broader goals of financial and societal advancement.
