# Scripts
Repository is a collection of small, usually single-file projects. Most of them have no actual purpose or application. Created while learning something or experimenting.

## Minifetch
    Neofetch clone for debian based systems:
![minifetch.png](assets/minifetch.png)

## Red ball tracker
    Tracks red ball on the video and draws bounding box:
![red-ball.png](assets/red-ball.png)

## RPS Markov
    "Rock, Paper, Scissors" vs Markov model that adjusts to your choices

## Crypto regression model
https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory

    CLI scripts to generate crypto dataset like:
    $ generate_dataset.py -o dataset.csv monero.csv doge.csv bitcoin.csv

    And then build Linear Regression like:
    $ run_regression --data dataset.csv --observation_window 3 --predicted_currency monero

    Note: of course it is just a statistical model and it won't predict actual spikes:
![crypto-prediction.png](assets/crypto-prediction.png)

## Coin detection and counting
    Detects, counts and separates coins:
![coin-detection-plot.png](assets/coin-detection-plot.png)