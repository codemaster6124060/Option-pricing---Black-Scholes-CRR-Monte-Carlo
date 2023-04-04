# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:54:49 2023

@author: banik
"""

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from itertools import chain
# plt.style.use('fast')
from matplotlib import cm
from matplotlib.ticker import LinearLocator


asset = yf.Ticker("SPY")
expirations = asset.options

def option_chains(ticker):
    """
    """
    chains = pd.DataFrame()
    
    for expiration in expirations:
        # tuple of two dataframes
        opt = asset.option_chain(expiration)
        
        calls = opt.calls
        calls['optionType'] = "call"
        
        puts = opt.puts
        puts['optionType'] = "put"
        
        chain = pd.concat([calls, puts])
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        
        chains = pd.concat([chains, chain])
    
    chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
    
    return chains
    # return calls

options = option_chains("SPY")
pd.set_option('display.max_columns', None)
print(options)
puts = options[options["optionType"] == "put"]
puts.expiration = pd.to_datetime(puts.expiration).dt.date


# Next, pick an expiration so you can plot the implied vola.
# print the expirations
set(puts.expiration)

# select an expiration to plot
for i in range(len(puts.expiration.unique())):
    puts_at_expiry = puts[puts["expiration"] ==  puts.expiration.unique()[i]]
    puts_at_expiry[["strike", "impliedVolatility"]].set_index("strike").plot(
    title=f"Implied Volatility Skew of put options expiring at {expirations[i]}", figsize=(7, 4))

#-----------------------------------------------------------------------------#
# Black Scholes Put Option Pricing
#-----------------------------------------------------------------------------#
from scipy.stats import norm
S = float(asset.history("1d","1d").Close)
r = 0.02 
q = 0
N_prime = norm.pdf
N = norm.cdf

simmat = np.zeros((len(puts_at_expiry),1))

def black_scholes_put(S, r):
    '''
    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :param q: dividend yield
    :return: put price
    '''
    for i in range(len(puts_at_expiry)):
        K = puts_at_expiry.strike[i]
        T = puts_at_expiry.daysToExpiration[i]/252 
        put_price = puts_at_expiry.lastPrice[i]
        sigma = puts_at_expiry.impliedVolatility[i]*np.sqrt(T)#/np.sqrt(252) # volatility
    
        ###standard black-scholes formula
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
        put = N(-d2)* K * np.exp(-r * T) - S* np.exp(-q*T) * N(-d1)
        simmat[i,0] = put
    BS_Put = pd.DataFrame(simmat,columns=["Black-Scholes Put Price"])
    # BS_Put['Expirations'] = puts.expiration.unique()
    return BS_Put    
black_scholes_put(S, r)

#-----------------------------------------------------------------------------#
# Bionial Option Tree Pricing
#-----------------------------------------------------------------------------#
option_type = list(options["optionType"].unique())[1]
steps = 200 
sigma = puts_at_expiry.impliedVolatility
K = puts_at_expiry.strike
T = puts_at_expiry.daysToExpiration[i]/252

def binomial_tree_price(S, K, r, sigma, T, steps, option_type):
    """
    Calculates the price of an option using the Cox, Rubenstein, and Ross binomial tree model.
    
    Parameters:
    S (float): current price of the underlying asset
    K (float): strike price of the option
    r (float): risk-free interest rate
    sigma (float): volatility of the underlying asset
    T (float): time to expiration of the option (in years)
    N (int): number of time steps in the binomial tree
    option_type (str): "call" or "put"
    
    Returns:
    float: the price of the option
    """

    
    stock_prices = np.zeros((steps + 1, steps + 1))
    option_prices = np.zeros((steps + 1, steps + 1))
    CRR_price = np.zeros((len(puts_at_expiry),1))
    
    for n_c in range(len(puts_at_expiry)-1):
        for i in range(steps + 1):
            for j in range(i + 1):
                delta_t = T / steps
                u = np.exp(sigma * np.sqrt(delta_t))[n_c]
                d = 1 / u
                p = (np.exp(r * delta_t) - d) / (u - d)
                stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)
        
        if option_type == "call":
            option_prices[:, steps] = np.maximum(stock_prices[:, steps] - K[n_c], 0)
        else:
            option_prices[:, steps] = np.maximum(K[n_c] - stock_prices[:, steps], 0)
        
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_prices[j, i] = np.exp(-r * delta_t) * (p * option_prices[j, i+1] + (1 - p) * option_prices[j+1, i+1])
                if option_type == "call":
                    option_prices[j, i] = np.maximum(option_prices[j, i], stock_prices[j, i] - K[n_c])
                else:
                    option_prices[j, i] = np.maximum(option_prices[j, i], K[n_c] - stock_prices[j, i])
        
        CRR_price[n_c,0] = option_prices[0, 0]
    CRR_put_price = pd.DataFrame()
    CRR_put_price["Contract Symbol"] = puts_at_expiry.contractSymbol
    CRR_put_price["Strike Price ($)"] = puts_at_expiry.strike
    CRR_put_price["Bid Price ($)"] = puts_at_expiry.bid
    CRR_put_price["Ask Price ($)"] = puts_at_expiry.ask
    CRR_put_price["Implied Volatility"] = puts_at_expiry.impliedVolatility
    CRR_put_price["Expiration"] = puts_at_expiry.expiration
    CRR_put_price["Last Price"] = puts_at_expiry.lastPrice
    CRR_put_price["Black-Scholes Put Price"] = black_scholes_put(S, r)
    CRR_put_price["CRR Option price"] = pd.DataFrame(CRR_price)
    return round(CRR_put_price,2)

American_put_option = binomial_tree_price(S, K, r, sigma, T, steps, option_type)

#-----------------------------------------------------------------------------#
# Monte Carlo Simulation for put option
#-----------------------------------------------------------------------------#
num_simulations = 500
num_time_steps = 200
drift = r
dt = T / steps

# Simulation
def MonteCarloSimulation():
    discount_factor = np.exp(-r * dt)
    simmat = np.zeros((len(puts_at_expiry),1))
    for j in range(len(puts_at_expiry)-1):
        # Simulate stock prices
        S0 = np.zeros((num_simulations, num_time_steps+1))
        S0[:, 0] = S
        for i in range(num_time_steps):
            eps = np.random.normal(0, 1, size=num_simulations)
            S0[:, i+1] = S0[:, i] * np.exp((r - 0.5*sigma[j]**2)*dt + sigma[j]*np.sqrt(dt)*eps)
        
        # Calculate option payoffs
        put_payoff = np.maximum(K[j]- S0, 0)
        
        # Calculate option values using Monte Carlo simulation
        put_value = np.zeros(num_simulations)
        for i in range(num_time_steps-1, -1, -1):
            exercise_value = put_payoff[:, i+1]
            continuation_value = discount_factor * put_value + exercise_value
            put_price = np.maximum(put_payoff[:, i], continuation_value)
        
        # Calculate option price
        put_price = np.mean(put_price)
    # print("American put option price:", put_price)
        simmat[j,0] = put_price
        put_price = pd.DataFrame(simmat)
    return round(put_price,2)

American_put_option["Monte Carlo Simulation Price"] = MonteCarloSimulation()
print(American_put_option)

#-----------------------------------------------------------------------------#
# Root Mean Squared Error for each pricing methods
#-----------------------------------------------------------------------------#
rmse = pd.DataFrame()
rmse["Contract Symbol"] = American_put_option["Contract Symbol"]
rmse["RMSE of BSM"] = np.sqrt(np.square(
    American_put_option["Black-Scholes Put Price"] 
    - American_put_option["Black-Scholes Put Price"].mean())/len(American_put_option))
rmse["RMSE of CRR"] = np.sqrt(np.square(
    American_put_option["CRR Option price"] 
    - American_put_option["CRR Option price"].mean())/len(American_put_option))
rmse["RMSE of Monte Carlo"] = np.sqrt(np.square(
    American_put_option["Monte Carlo Simulation Price"] 
    - American_put_option["Monte Carlo Simulation Price"].mean())/len(American_put_option))
rmse

#-----------------------------------------------------------------------------#
# Vary the number of steps in the CRR model and the number of simulations used 
# for the Monte Carlo method.
#-----------------------------------------------------------------------------#
# Change number of steps from 200 to 300
steps = 500
modified_pricing = binomial_tree_price(S, K, r, sigma, T, steps, option_type)
# Change number of simulations from 500 to 1000
num_simulations = 1000
modified_pricing["Monte Carlo Simulation Price"] = MonteCarloSimulation()
print(modified_pricing)


rmse["Modified RMSE of CRR"] = np.sqrt(np.square(
    modified_pricing["CRR Option price"] 
    - modified_pricing["CRR Option price"].mean())/len(modified_pricing))
rmse["Modified RMSE of Monte Carlo"] = np.sqrt(np.square(
    modified_pricing["Monte Carlo Simulation Price"] 
    - modified_pricing["Monte Carlo Simulation Price"].mean())/len(modified_pricing))
rmse

