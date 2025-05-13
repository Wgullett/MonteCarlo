import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N_SIM = 5000
NUMBER_OF_DISPLAYED_SIMS = 10


def run_monte_carlo(returns, simulations):
    """
    Run Monte Carlo simulation to generate random portfolios

    """
    num_assets = returns.shape[1]
    asset_tickers = returns.columns
    mean_returns = returns.mean()
    cov_matrix = returns.cov() # Covariance matrix of returns (DOBULE CHECK)
    time_horizon = returns.shape[0] # Number of trading days in a year

    results = np.zeros((simulations, time_horizon, num_assets))
    portfolio_returns = np.zeros((simulations, time_horizon))
    final_portfolio_values = np.zeros((simulations))

    inital_portfolio_value = 1  # Initial portfolio value

    for i in range(simulations):\
        # Generate random weights
        weights = np.random.random(num_assets)  
        weights /= np.sum(weights)

        #Simulate Future Asset Prices
        asset_paths = np.zeros((time_horizon, num_assets))
        asset_paths[0, :] = inital_portfolio_value  # Start with an initial value of 1 for each asset

        for t in range(1, time_horizon):
            # Generate random returns based on the mean and covariance matrix
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            asset_paths[t, :] = asset_paths[t - 1, :] * (1 + random_returns)

        results[i, :, :] = asset_paths

        #Calculate Porftfolio Values over time
        portfolio_values = np.sum(asset_paths, axis=1)
        portfolio_returns[i, :] = (portfolio_values / inital_portfolio_value) - 1
        final_portfolio_values[i] = portfolio_values[-1]

    return portfolio_returns, final_portfolio_values

def analzye_sim_results(returns, final_values, initial_investment=10000):
    """
    Analyzes the results of the Monte Carlo simulations.

    Args:
        portfolio_returns (pd.DataFrame): DataFrame of simulated portfolio returns.
        final_portfolio_values (np.ndarray): Array of final portfolio values for each simulation.
        initial_investment (float): The initial investment amount.

    Returns:
        dict: A dictionary containing key statistics from the simulations.
    """
    confidence_intervals = np.percentile(final_portfolio_values * initial_investment, [5, 50, 95])
    expected_final_value = np.mean(final_portfolio_values) * initial_investment
    probability_of_loss = (final_portfolio_values < 1).sum() / len(final_portfolio_values)

    results_analysis = {
        'Expected Final Value': expected_final_value,
        '5th Percentile': confidence_intervals[0],
        '50th Percentile (Median)': confidence_intervals[1],
        '95th Percentile': confidence_intervals[2],
        'Probability of Loss (Final Value < Initial)': probability_of_loss * 100,
    }
    return results_analysis

# Get historical data for Apple
aapl = yf.Ticker("AAPL")
historical_data_appl = aapl.history(period="5y") # Get data for the last 5 years
print("Historical data for Apple:")
print(historical_data_appl.head())

# Get historical data for Microsoft
msft = yf.Ticker("MSFT")
historical_data_msft = msft.history(period="5y") # Get data for the last 5 years
print("Historical data for Microsoft:")
print(historical_data_msft.head())

# Get historical data for Amazon
amzn = yf.Ticker("AMZN")
historical_data_amzn = amzn.history(period="5y") # Get data for the last 5 years
print("Historical data for Amazon:")
print(historical_data_amzn.head())

# Get historical data for Johnson & Johnson
jnj = yf.Ticker("JNJ")
historical_data_jnj = jnj.history(period="5y") # Get data for the last 5 years
print("Historical data for Johnson & Johnson:")
print(historical_data_jnj.head())

# Get historical data for JPMorgan Chase
jpm = yf.Ticker("JPM")
historical_data_jpm = jpm.history(period="5y") # Get data for the last 5 years
print("Historical data for JPMorgan Chase:")
print(historical_data_jpm.head())


# Load and Seleect Closing Prices
aapl_prices = historical_data_appl.reset_index()[['Date', 'Close']].rename(columns={'Close': 'AAPL'}).set_index('Date')
msft_prices = historical_data_msft.reset_index()[['Date', 'Close']].rename(columns={'Close': 'MSFT'}).set_index('Date')
amzn_prices = historical_data_amzn.reset_index()[['Date', 'Close']].rename(columns={'Close': 'AMZN'}).set_index('Date')
jnj_prices = historical_data_jnj.reset_index()[['Date', 'Close']].rename(columns={'Close': 'JNJ'}).set_index('Date')
jpm_prices = historical_data_jpm.reset_index()[['Date', 'Close']].rename(columns={'Close': 'JPM'}).set_index('Date')

# Merge Price DataFrames
portfolio_prices = pd.concat([aapl_prices, msft_prices, amzn_prices, jnj_prices, jpm_prices], axis=1, join='inner')

print("Merged Portfolio Prices:")
print(portfolio_prices.head())

# Calculate Daily Returns
# Return_t = (Price_t - Price_{t-1}) / Price_{t-1}
daily_returns = portfolio_prices.pct_change().dropna()

print("Daily Returns:")
print(daily_returns.head())

# Calculate Covariance Matrix
mean_returns = daily_returns.mean()
volatility = daily_returns.std()
correlation_matrix = daily_returns.corr()

print("Mean Returns:\n", mean_returns)
print("\nVolatility:\n", volatility)
print("\nCorrelation Matrix:\n", correlation_matrix)

# Monte Carlo Simulation
# Simulate 5000 portfolios
sim_returns, final_portfolio_values = run_monte_carlo(daily_returns, N_SIM)
analzye_sim_results = analzye_sim_results(daily_returns, final_portfolio_values, initial_investment=10000)
print("Monte Carlo Simulation Results:")
for key, value in analzye_sim_results.items():
     print(f"{key}: {value:.2f}%" if key.startswith("Probability") else f"{key}: ${value:.2f}")

#Figure 1 - Plotting the Monte Carlo Simulations
plots = []
plt.figure(figsize=(10, 6))
for i in range(min(NUMBER_OF_DISPLAYED_SIMS, N_SIM)):  # Plot a few simulations
    plots = plt.plot(sim_returns[i], label=f"Simulation {i+1}")
plt.title('Monte Carlo Simulation of Portfolio Returns')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Return (relative to initial)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#Figure 2 - Plotting Distribution of Final Portfolio Values
plt.figure(figsize=(8, 6))
plt.hist(final_portfolio_values * 10000, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(analzye_sim_results['Expected Final Value'], color='r', linestyle='dashed', linewidth=1, label=f"Expected: ${analzye_sim_results['Expected Final Value']:.2f}")
plt.axvline(analzye_sim_results['5th Percentile'], color='y', linestyle='dashed', linewidth=1, label=f"5th Percentile: ${analzye_sim_results['5th Percentile']:.2f}")
plt.axvline(analzye_sim_results['95th Percentile'], color='y', linestyle='dashed', linewidth=1, label=f"95th Percentile: ${analzye_sim_results['95th Percentile']:.2f}")
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()