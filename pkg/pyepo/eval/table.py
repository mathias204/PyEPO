import re

import pandas as pd


translation = {
    r'$\hat{z}^{kNN}_N(x)$': r'$\hat{z}^{kNN}_N$',
    r'$\hat{z}^{LOESS}_N(x)$': r'$\hat{z}^{LOESS}_N$',
    r'$\hat{z}^{KR}_N(x)$': r'$\hat{z}^{KR}_N$',
    r'$\hat{z}^{Rec.-KR}_N(x)$': r'$\hat{z}^{Rec.KR}_N$',
    r'$\hat{z}^{RF}_N(x)$': r'$\hat{z}^{RF}_N$',
    r'$\hat{z}^{CART}_N(x)$': r'$\hat{z}^{CART}_N$',
    r'$\hat{z}^{SAA}_N(x)$': r'$\hat{z}^{SAA}_N$',
    r'$\hat{z}^{DER}_N(x)$': r'$\hat{z}^{DER}_N$',
    r'$\hat{z}^{SPO+}_N(x)$': r'$\hat{z}^{SPO+}_N$',
    r'$z^{SPO+}(x)$': r'$z^{SPO+}$',
    r'$z^{SFGE}(x)$': r'$z^{SFGE}$',
    r'$z^{MSE}(x)$': r'$z^{MSE}$',
    
}


def parse_tensor_string(val):
    if isinstance(val, str) and "tensor" in val:
        # Extract the number inside the parentheses
        match = re.search(r"tensor\((.*?)\)", val)
        if match:
            return float(match.group(1))
    return float(val)
    

def report_model_statistics(csv_path, wpp_only=False, sort_on='mean'):
    """
    Calculates and prints summary statistics for relative regret across models.
    If wpp_only is True, it strictly reports the statistics for the WPP method.
    """
    # Load the data
    df = pd.read_csv(csv_path)

    # Filter out the SAA model to match the plotting behavior
    saa_model = r'$\hat{z}^{SAA}_N(x)$'
    if saa_model in df['model_name'].values:
        df = df[df['model_name'] != saa_model]
        
    # Multiply by 100 to convert to percentages
    df['result_pct'] = df['result'].apply(parse_tensor_string) * 100

    # Translate model names early so we can filter predictably
    if translation:
        df['model_name'] = df['model_name'].apply(lambda x: translation.get(x, x))

    # --- NEW: Filter for WPP only ---
    if wpp_only:
        # Uses case-insensitive matching to find 'wpp' or 'weighted predictive' 
        # in case the translation dictionary slightly alters the name
        mask = df['model_name'].str.contains('wpp|weighted predictive', case=False, na=False)
        df = df[mask]
        
        if df.empty:
            print("⚠️ No model matching 'WPP' or 'Weighted Predictive Prescription' was found.")
            return

    # Calculate statistics
    stats = df.groupby('model_name')['result_pct'].agg(
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max'
    ).reset_index()

    stats = stats.sort_values(by=sort_on, ascending=True)
    
    # Formatters for clean percentage outputs
    formatters = {
        'mean': '{:>6.2f}%'.format,
        'median': '{:>6.2f}%'.format,
        'std': '{:>6.2f}%'.format,
        'min': '{:>6.2f}%'.format,
        'max': '{:>6.2f}%'.format
    }

    # --- Print Outputs ---
    if wpp_only:
        print("🎯 Statistics for Weighted Predictive Prescription (WPP) 🎯")
        print("-" * 65)
        print(stats.to_string(index=False, formatters=formatters))
    else:
        print("🏆 Top 3 Best Performing Models (Lowest Mean Relative Regret) 🏆")
        print("-" * 65)
        
        top_3 = stats.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), start=1):
            print(f"{i}. {row['model_name']:<15} | Mean: {row['mean']:>5.2f}% | Std Dev: {row['std']:>5.2f}%")
            
        print("\n📊 Full Statistical Summary (Sorted by Mean) 📊")
        print("-" * 65)
        print(stats.to_string(index=False, formatters=formatters))

# Example usage:
# report_model_statistics('path/to/your/data.csv', translation=translation)