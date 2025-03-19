import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as plt
import seaborn as sns
from scipy.stats import norm

# Black-Scholes Pricing Model Function
def black_scholes(S, K, T, r, sigma, option_type):
    """Calculate the option price using the Black-Scholes model."""
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose either 'call' or 'put'.")
    
    return price, d1, d2

# Greek Calculations
def calculate_greeks(S, K, T, r, sigma, option_type):
    """Calculate the Greeks for the Black-Scholes model."""
    price, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    
    # Delta
    delta = norm.cdf(d1) if option_type.lower() == 'call' else -norm.cdf(-d1)
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type.lower() == 'call' else -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    # Rho
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type.lower() == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return delta, gamma, theta, vega, rho

# Function to create the sidebar UI elements
def get_user_inputs():
    """Collect user inputs from the Streamlit sidebar."""
    with st.sidebar:
        st.title("Option Parameters")
        current_stock_price = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.01)
        interest_rate = st.number_input("Risk-Free Interest Rate (r)", value=0.05, min_value=0.0, max_value=1.0)
        time = st.number_input("Time to Maturity (T) in Years", value=1.0, min_value=0.01)
        original_volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01)
        
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        # Spot price range configuration
        st.header("Heat Map Parameters")
        spot_price_min = st.number_input("Minimum Spot Price", value=current_stock_price * 0.5, min_value=0.01)
        spot_price_max = st.number_input("Maximum Spot Price", value=current_stock_price * 1.5, min_value=0.01)
        # Volatility slider to create a range around original volatility
        min_volatility = original_volatility * 0.5
        max_volatility = original_volatility * 1.5
        volatility_range = st.slider("Volatility Range (σ)", min_value=min_volatility, max_value=max_volatility, value=(min_volatility, max_volatility), step=0.01)
    
    return current_stock_price, strike_price, time, interest_rate, original_volatility, option_type, spot_price_min, spot_price_max, volatility_range

# Function to plot the Greek letters
def plot_greeks(S, K, T, r, sigma, option_type):
    """Plot the Greeks dynamically based on user input."""
    # Delta vs Stock Price
    stock_prices = np.linspace(S * 0.5, S * 1.5, 100)
    delta_values = [calculate_greeks(sp, K, T, r, sigma, option_type)[0] for sp in stock_prices]
    
    # Gamma vs Stock Price
    gamma_values = [calculate_greeks(sp, K, T, r, sigma, option_type)[1] for sp in stock_prices]
    
    # Theta vs Time to Maturity
    times = np.linspace(0.01, 5.0, 100)
    theta_values = [calculate_greeks(S, K, t, r, sigma, option_type)[2] for t in times]
    
    # Vega vs Volatility
    volatilities = np.linspace(0.01, 2.0, 100)
    vega_values = [calculate_greeks(S, K, T, r, vol, option_type)[3] for vol in volatilities]
    
    # Rho vs Interest Rate
    interest_rates = np.linspace(0.0, 0.1, 100)
    rho_values = [calculate_greeks(S, K, T, ir, sigma, option_type)[4] for ir in interest_rates]
    
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    
    # Delta
    axs[0, 0].plot(stock_prices, delta_values, label="Delta")
    axs[0, 0].set_title("Delta vs Stock Price")
    axs[0, 0].set_xlabel("Stock Price (S)")
    axs[0, 0].set_ylabel("Delta")
    axs[0, 0].grid(True)
    
    # Gamma
    axs[0, 1].plot(stock_prices, gamma_values, label="Gamma", color='orange')
    axs[0, 1].set_title("Gamma vs Stock Price")
    axs[0, 1].set_xlabel("Stock Price (S)")
    axs[0, 1].set_ylabel("Gamma")
    axs[0, 1].grid(True)
    
    # Theta
    axs[1, 0].plot(times, theta_values, label="Theta", color='green')
    axs[1, 0].set_title("Theta vs Time to Maturity")
    axs[1, 0].set_xlabel("Time to Maturity (T)")
    axs[1, 0].set_ylabel("Theta")
    axs[1, 0].grid(True)
    
    # Vega
    axs[1, 1].plot(volatilities, vega_values, label="Vega", color='red')
    axs[1, 1].set_title("Vega vs Volatility")
    axs[1, 1].set_xlabel("Volatility (σ)")
    axs[1, 1].set_ylabel("Vega")
    axs[1, 1].grid(True)
    
    # Rho
    axs[2, 0].plot(interest_rates, rho_values, label="Rho", color='purple')
    axs[2, 0].set_title("Rho vs Interest Rate")
    axs[2, 0].set_xlabel("Interest Rate (r)")
    axs[2, 0].set_ylabel("Rho")
    axs[2, 0].grid(True)
    
    # Hide empty subplot
    axs[2, 1].axis('off')
    
    # Adjusting layout to prevent label overlap
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot the heatmap (correlation matrix)

def plot_option_price_heatmap(S, K, T, r, sigma, spot_price_min, spot_price_max, volatility_range):
    """
    Plot a heatmap to visualize call and put prices based on volatility (y-axis) and spot prices (x-axis)
    """

    # Create the range for spot prices and volatility
    spot_prices = np.linspace(spot_price_min, spot_price_max, 11)  # 20 spot prices
    volatilities = np.linspace(volatility_range[0], volatility_range[1], 11)  # 20 volatility values
    
    # Initialize matrices for storing the option prices (call and put)
    call_prices = np.zeros((len(volatilities), len(spot_prices)))  # Adjusted for volatility as y-axis
    put_prices = np.zeros((len(volatilities), len(spot_prices)))   # Adjusted for volatility as y-axis
    
    # Calculate option prices for each combination of volatility and spot price
    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            call_prices[i, j], _, _ = black_scholes(spot, K, T, r, vol, 'call')
            put_prices[i, j], _, _ = black_scholes(spot, K, T, r, vol, 'put')
    
    # Plotting both heatmaps side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

    # Create the heatmap for call prices
    sns.heatmap(call_prices, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=[f'{s:.2f}' for s in spot_prices], 
                yticklabels=[f'{v:.2f}' for v in volatilities], cbar_kws={'label': 'Call Option Price'}, ax=ax1,
                annot_kws={"size": 10, "weight": 'bold', "color": 'black'})
    ax1.set_xlabel("Spot Price (S)", fontsize=12)
    ax1.set_ylabel("Volatility (σ)", fontsize=12)
    ax1.set_title("Call Option Price Heatmap", fontsize=16)

    # Create the heatmap for put prices
    sns.heatmap(put_prices, annot=True, fmt=".2f", cmap="RdYlBu", xticklabels=[f'{s:.2f}' for s in spot_prices], 
                yticklabels=[f'{v:.2f}' for v in volatilities], cbar_kws={'label': 'Put Option Price'}, ax=ax2,
                annot_kws={"size": 10, "weight": 'bold', "color": 'black'})
    ax2.set_xlabel("Spot Price (S)", fontsize=12)
    ax2.set_ylabel("Volatility (σ)", fontsize=12)
    ax2.set_title("Put Option Price Heatmap", fontsize=16)

    # Adjust layout to prevent overlap of axes labels
    plt.tight_layout()
    st.pyplot(fig)

# Main function
def main():
    # Title for the webpage
    st.title("Option Pricing Analysis using Black-Scholes Model")
    
    # Get user inputs
    current_stock_price, strike_price, time, interest_rate, original_volatility, option_type, spot_price_min, spot_price_max, volatility_range = get_user_inputs()

    # Section 1: Display Call and Put Option Prices
    st.header("1. Call and Put Option Prices")
    col1, col2 = st.columns(2)  # Create two columns with space between them

    with col1:
        # Display Call Option in a large box
        call_price, _, _ = black_scholes(current_stock_price, strike_price, time, interest_rate, original_volatility, 'call')
        st.markdown(f"<div style='background-color:#F01E2C; padding:20px; border-radius:10px; margin-bottom:20px;'><h2 style='font-size:32px; text-align:center;'>Call Option: ${call_price:.2f}</h2></div>", unsafe_allow_html=True)
    
    with col2:
        # Display Put Option in a large box
        put_price, _, _ = black_scholes(current_stock_price, strike_price, time, interest_rate, original_volatility, 'put')
        st.markdown(f"<div style='background-color:#1C2951; padding:20px; border-radius:10px; margin-bottom:20px;'><h2 style='font-size:32px; text-align:center;'>Put Option: ${put_price:.2f}</h2></div>", unsafe_allow_html=True)

    # Section 2: Display Greeks Analysis
    st.header("2. Greeks Analysis")
    plot_greeks(current_stock_price, strike_price, time, interest_rate, original_volatility, option_type)
    
    # Section 3: Display Option Price Heatmap
    st.header("3. Option Price Heatmap")
    plot_option_price_heatmap(current_stock_price, strike_price, time, interest_rate, original_volatility, spot_price_min, spot_price_max, volatility_range)

if __name__ == "__main__":
    main()


