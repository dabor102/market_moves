import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

class AdvancedOptionsAnalyzer:
    """
    A class to fetch, analyze, and visualize options data with a focus on
    market maker positioning and sophisticated metrics like Open Interest,
    Implied Volatility, and Gamma Exposure (GEX).
    """

    def __init__(self, ticker_symbol):
        """
        Initializes the analyzer with a stock ticker symbol.

        Args:
            ticker_symbol (str): The stock ticker symbol (e.g., 'SPY', 'CRWD').
        """
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker_obj = yf.Ticker(self.ticker_symbol)
        self.spot_price = self._get_spot_price()
        print(f"Initialized Analyzer for {self.ticker_symbol} | Current Spot Price: ${self.spot_price:.2f}")

    def _get_spot_price(self):
        """Fetches the most recent spot price of the underlying asset."""
        try:
            hist = self.ticker_obj.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1]
            # Fallback for when history fails
            data = self.ticker_obj.fast_info
            return data.get('last_price', 0)
        except Exception as e:
            print(f"Error fetching spot price for {self.ticker_symbol}: {e}")
            return 0

    def get_options_data(self, expiration_date):
        """
        Fetches and preprocesses options data for a given expiration date.
        It calculates greeks if they are not provided by the API.

        Args:
            expiration_date (str): The expiration date in 'YYYY-MM-DD' format.

        Returns:
            tuple: A tuple of (calls_df, puts_df). Returns (None, None) on failure.
        """
        try:
            option_chain = self.ticker_obj.option_chain(expiration_date)
            calls = option_chain.calls
            puts = option_chain.puts

            if calls.empty or puts.empty:
                print(f"No options data found for {expiration_date}.")
                return None, None

            # Ensure essential columns exist and are numeric
            for df in [calls, puts]:
                for col in ['openInterest', 'volume', 'impliedVolatility', 'strike']:
                    if col not in df.columns:
                        print(f"Warning: Column '{col}' not found. Filling with 0.")
                        df[col] = 0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # YFinance sometimes omits greeks. If 'gamma' is missing, GEX can't be calculated.
            if 'gamma' not in calls.columns or 'gamma' not in puts.columns:
                print("\nWarning: 'gamma' data is not available from the API for this ticker.")
                print("GEX-related calculations will be skipped.")

            print(f"Successfully fetched {len(calls)} calls and {len(puts)} puts for {expiration_date}")
            return calls, puts
        except Exception as e:
            print(f"An error occurred fetching options data for {expiration_date}: {e}")
            return None, None

    def get_closest_expiration_date(self):
        """Gets the next available monthly expiration date."""
        try:
            expirations = self.ticker_obj.options
            if not expirations:
                return None
            
            # Find the next standard third Friday of the month, a common monthly expiration
            today = datetime.now()
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                if exp_date > today:
                    return exp_str # Return the first available future expiration
            return expirations[-1] if expirations else None # Fallback to last available
        except Exception as e:
            print(f"Could not retrieve expiration dates: {e}")
            return None

    def analyze_options_overview(self, calls_df, puts_df):
        """
        Prints a sophisticated overview of the options landscape, incorporating
        Open Interest, Volume, and the Put/Call ratio.
        """
        print("\n" + "="*50)
        print("         Options Landscape Overview")
        print("="*50)

        total_call_oi = calls_df['openInterest'].sum()
        total_put_oi = puts_df['openInterest'].sum()
        total_call_vol = calls_df['volume'].sum()
        total_put_vol = puts_df['volume'].sum()

        print(f"\nTotal Call OI: {total_call_oi:,.0f} | Total Put OI: {total_put_oi:,.0f}")
        print(f"Total Call Vol: {total_call_vol:,.0f} | Total Put Vol: {total_put_vol:,.0f}")

        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else float('inf')
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else float('inf')
        print(f"\nPut/Call Ratio (by Open Interest): {pcr_oi:.2f}")
        print(f"Put/Call Ratio (by Volume): {pcr_vol:.2f}")

        calls_df['oi_vol_product'] = calls_df['openInterest'] * calls_df['volume']
        puts_df['oi_vol_product'] = puts_df['openInterest'] * puts_df['volume']

        top_calls = calls_df.nlargest(5, 'oi_vol_product')[['strike', 'openInterest', 'volume', 'impliedVolatility']]
        top_puts = puts_df.nlargest(5, 'oi_vol_product')[['strike', 'openInterest', 'volume', 'impliedVolatility']]

        print("\n--- Key Option Strikes (High OI & Volume) ---")
        print("\nTop 5 Calls:")
        print(top_calls.to_string())
        print("\nTop 5 Puts:")
        print(top_puts.to_string())
        print("="*50 + "\n")


    def plot_volume_oi_profile(self, calls_df, puts_df, expiration_date, strike_range_pct=0.20):
        """
        Plots a histogram of Open Interest and Volume with cumulative profile lines.
        Calls are positive, Puts are negative. Cumulative lines use a secondary axis.
        """
        min_strike = self.spot_price * (1 - strike_range_pct)
        max_strike = self.spot_price * (1 + strike_range_pct)

        calls = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
        puts = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

        call_data = calls.groupby('strike')[['openInterest', 'volume']].sum()
        put_data = puts.groupby('strike')[['openInterest', 'volume']].sum() * -1

        all_strikes = sorted(list(set(call_data.index.tolist() + put_data.index.tolist())))
        profile_df = pd.DataFrame(index=all_strikes)
        profile_df = profile_df.join(call_data.rename(columns={'openInterest': 'call_oi', 'volume': 'call_vol'}))
        profile_df = profile_df.join(put_data.rename(columns={'openInterest': 'put_oi', 'volume': 'put_vol'}))
        profile_df.fillna(0, inplace=True)
        
        # --- Calculate Cumulative Profiles ---
        profile_df['total_oi'] = profile_df['call_oi'] + profile_df['put_oi'].abs()
        profile_df['total_vol'] = profile_df['call_vol'] + profile_df['put_vol'].abs()
        profile_df['cum_oi'] = profile_df['total_oi'].cumsum()
        profile_df['cum_vol'] = profile_df['total_vol'].cumsum()
        
        # --- Plotting ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))

        # --- Plot Bars (Primary Y-Axis) ---
        bar_width = 0.8 * (profile_df.index[1] - profile_df.index[0] if len(profile_df.index) > 1 else 1)
        ax.bar(profile_df.index, profile_df['call_oi'], width=bar_width, color='blue', label='Call OI')
        ax.bar(profile_df.index, profile_df['put_oi'], width=bar_width, color='red', label='Put OI')
        ax.bar(profile_df.index, profile_df['call_vol'], width=bar_width, color='cyan', alpha=0.7, label='Call Volume')
        ax.bar(profile_df.index, profile_df['put_vol'], width=bar_width, color='orange', alpha=0.7, label='Put Volume')

        # --- Formatting for Primary Y-Axis ---
        ax.set_title(f'{self.ticker_symbol} Open Interest & Volume Profile\nExp: {expiration_date} | Spot: ${self.spot_price:.2f}', color='white')
        ax.set_ylabel('Contracts per Strike (Puts are negative)', color='deepskyblue')
        ax.set_xlabel('Strike Price', color='white')
        ax.tick_params(axis='y', labelcolor='deepskyblue')
        ax.tick_params(axis='x', colors='white')
        formatter = mtick.FuncFormatter(lambda x, p: f'{abs(x):,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--', alpha=0.2, axis='y')

        # --- Plot Cumulative Lines (Secondary Y-Axis) ---
        ax2 = ax.twinx()
        ax2.plot(profile_df.index, profile_df['cum_oi'], color='yellow', linestyle='-', linewidth=2, label='Cumulative OI')
        ax2.plot(profile_df.index, profile_df['cum_vol'], color='gold', linestyle='--', linewidth=2, label='Cumulative Volume')

        # --- Formatting for Secondary Y-Axis ---
        ax2.set_ylabel('Cumulative Contracts', color='yellow')
        ax2.tick_params(axis='y', labelcolor='yellow')
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # --- Combined Formatting & Legend ---
        ax.axhline(0, color='white', linestyle='-', linewidth=0.7)
        ax.axvline(self.spot_price, color='white', linestyle=':', linewidth=1.5, label=f'Spot Price: ${self.spot_price:.2f}')
        
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()


    def plot_gex_profile(self, calls_df, puts_df, expiration_date, strike_range_pct=0.20):
        """
        Calculates and plots the Gamma Exposure (GEX) profile.
        """
        if 'gamma' not in calls_df.columns or 'gamma' not in puts_df.columns:
            print("Cannot plot GEX Profile because 'gamma' data is missing.")
            return

        min_strike = self.spot_price * (1 - strike_range_pct)
        max_strike = self.spot_price * (1 + strike_range_pct)
        calls = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)].copy()
        puts = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)].copy()

        calls['gex'] = calls['gamma'] * calls['openInterest'] * 100
        puts['gex'] = puts['gamma'] * puts['openInterest'] * 100 * -1

        gex_by_strike = pd.concat([calls.groupby('strike')['gex'].sum(), puts.groupby('strike')['gex'].sum()]).groupby('strike').sum()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['limegreen' if val > 0 else 'firebrick' for val in gex_by_strike.values]
        gex_by_strike.plot(kind='bar', ax=ax, color=colors)

        total_gex = gex_by_strike.sum()
        cumulative_gex = gex_by_strike.cumsum()
        
        try:
            zero_gex_level_idx = (np.abs(cumulative_gex)).idxmin()
            ax.axvline(x=zero_gex_level_idx, color='yellow', linestyle='--', linewidth=2, label=f'Gamma Flip: {zero_gex_level_idx:.0f}')
        except ValueError:
            pass

        ax.set_title(f'Gamma Exposure (GEX) for {self.ticker_symbol}\nExp: {expiration_date} | Total GEX: ${total_gex / 1e9:.2f}B', color='white')
        ax.set_ylabel('Gamma Exposure (Notional)', color='white')
        ax.set_xlabel('Strike Price', color='white')
        formatter = mtick.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M')
        ax.yaxis.set_major_formatter(formatter)
        ax.axhline(0, color='white', linestyle='-', linewidth=0.7)
        ax.axvline(x=self.spot_price, color='cyan', linestyle=':', linewidth=2, label=f'Spot Price: ${self.spot_price:.2f}')
        ax.tick_params(axis='x', colors='white', rotation=90)
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_iv_skew(self, calls_df, puts_df, expiration_date):
        """
        Plots the Implied Volatility (IV) skew or "smile".
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(calls_df['strike'], calls_df['impliedVolatility'], 'o-', label='Call IV', color='deepskyblue')
        ax.plot(puts_df['strike'], puts_df['impliedVolatility'], 'o-', label='Put IV', color='orangered')

        ax.set_title(f'Implied Volatility Skew for {self.ticker_symbol}\nExp: {expiration_date}', color='white')
        ax.set_xlabel('Strike Price', color='white')
        ax.set_ylabel('Implied Volatility', color='white')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.axvline(self.spot_price, color='yellow', linestyle='--', linewidth=1.5, label=f'Spot Price: ${self.spot_price:.2f}')
        
        atm_iv = calls_df.iloc[(calls_df['strike'] - self.spot_price).abs().argsort()[:1]]['impliedVolatility'].values[0]
        ax.axhline(atm_iv, color='grey', linestyle=':', linewidth=1, label=f'ATM IV: {atm_iv:.1%}')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        plt.tight_layout()
        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    target_ticker = 'TSLA' # Change to 'SPY', 'QQQ', 'CRWD', etc.
    
    # --- Initialization ---
    analyzer = AdvancedOptionsAnalyzer(target_ticker)
    
    if analyzer.spot_price == 0:
        print(f"Could not retrieve spot price for {target_ticker}. Exiting.")
        exit()

    # --- Fetch Data ---
    expiration = analyzer.get_closest_expiration_date()
    if not expiration:
        print(f"No expiration dates found for {target_ticker}. Exiting.")
        exit()
        
    calls, puts = analyzer.get_options_data(expiration)

    if calls is not None and puts is not None:
        # --- Perform Analysis & Generate Plots ---
        analyzer.analyze_options_overview(calls, puts)
        analyzer.plot_volume_oi_profile(calls, puts, expiration)
        analyzer.plot_iv_skew(calls, puts, expiration)
        analyzer.plot_gex_profile(calls, puts, expiration)

    else:
        print(f"Failed to retrieve or process options data for {target_ticker}.")