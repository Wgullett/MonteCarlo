# Portfolio Risk and Return Analysis using Monte Carlo Simulation

## Overview

This project implements a Monte Carlo simulation in Python to analyze the potential future performance and risk of a multi-asset stock portfolio. By leveraging historical stock price data obtained from Yahoo Finance via the `yfinance` library, the simulation generates numerous possible portfolio return trajectories over a specified time horizon. The analysis of these simulations provides insights into the expected final portfolio value, potential downside risk (probability of loss), and a range of likely outcomes presented through confidence intervals.

## Key Features

* **Data Acquisition:** Automatically fetches historical adjusted closing prices for a user-defined list of stock tickers using the `yfinance` library.
* **Return Calculation:** Computes daily percentage returns for each asset based on the historical price data.
* **Monte Carlo Simulation:** Runs a specified number of simulations, each with randomly generated portfolio weights, to model potential future portfolio returns based on the historical mean returns and covariance matrix.
* **Risk and Return Analysis:** Calculates key metrics from the simulation results, including:
    * Expected final portfolio value.
    * 5th, 50th (median), and 95th percentile confidence intervals for the final portfolio value.
    * Probability of the final portfolio value being less than the initial investment.
* **Visualization:** Generates Matplotlib plots to visualize:
    * A sample of individual simulated portfolio return paths over time.
    * A histogram showing the distribution of final portfolio values with overlaid confidence intervals and the expected value.

## Technologies Used

* Python 3
* Pandas
* NumPy
* Matplotlib
* yfinance

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Install required libraries:**
    ```bash
    pip install pandas numpy matplotlib yfinance
    ```

## Usage

1.  **Modify Parameters:** Open the Python script (e.g., `monte_carlo_portfolio.py`) and adjust the following parameters in the `if __name__ == '__main__':` block:
    * `tickers`: A list of the stock ticker symbols you want to include in your portfolio analysis (e.g., `['AAPL', 'MSFT', 'AMZN', 'JNJ', 'JPM']`).
    * `start_date`: The start date for the historical data (format: `YYYY-MM-DD`).
    * `end_date`: The end date for the historical data (format: `YYYY-MM-DD`).
    * `num_simulations`: The number of Monte Carlo simulations to run (increase for more robust results).
    * `time_horizon`: The number of trading days to simulate into the future (e.g., `252` for approximately one year).
    * `initial_investment`: The initial investment amount for the analysis (used for interpreting final values).

2.  **Run the script:**
    ```bash
    python monte_carlo_portfolio.py
    ```

3.  **View Results:** The script will print the analysis results in the console and display two Matplotlib plots showing the simulated portfolio returns and the distribution of final portfolio values.

## Example Output (With initial investment of $10,000)
Monte Carlo Simulation Results:
* Expected Final Value: $146648.48
* 5th Percentile: $60586.42
* 50th Percentile (Median): $128986.80
* 95th Percentile: $291225.51
* Probability of Loss (Final Value < Initial): 0.00%

![Figure_1_MonteCarlo](https://github.com/user-attachments/assets/43fe2f28-22d2-4ed4-9966-e4f2979c8c0c)

![Figure_2_MonteCarlo](https://github.com/user-attachments/assets/16382d49-4716-4129-a657-cbb1eeacdb6c)




