"""Visualization module for Exploratory Data Analysis, training metrics, and attention analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import FIGURES_DIR, TARGET_COLS, CLEANED_DATA_PATH

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_seasonality_and_trends(df: pd.DataFrame, save_path: Path = FIGURES_DIR / "01_yearly_trends_seasonality.png"):
    """Visual 1: Multi-year longitudinal trends & annual seasonality for all target variables."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    units = ["°C", "mbar", "%", "m/s"]

    for i, col in enumerate(TARGET_COLS):
        axes[i].plot(df.index, df[col], color=colors[i], alpha=0.75, linewidth=0.8, label=col)
        # 30-day moving average to reveal annual trends clearly
        ma_30d = df[col].rolling(window=24 * 30, center=True).mean()
        axes[i].plot(df.index, ma_30d, color="black", linewidth=1.5, label="30-Day Moving Average")
        axes[i].set_ylabel(f"{col} ({units[i]})")
        axes[i].legend(loc="upper right", framealpha=0.9)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    axes[0].set_title("Longitudinal Multi-Year Meteorological Trends (Jena Climate 2009–2016)", fontsize=14, fontweight="bold")
    axes[-1].set_xlabel("Date Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 1 to: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path = FIGURES_DIR / "02_correlation_heatmap.png"):
    """Visual 2: Correlation heatmap of selected targets and related thermodynamic indicators."""
    candidate_cols = TARGET_COLS + ["Tpot (K)", "Tdew (degC)", "VPmax (mbar)", "VPact (mbar)", "rho (g/m**3)"]
    available_cols = [c for c in candidate_cols if c in df.columns]
    corr = df[available_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Correlation Matrix of Meteorological Variables", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 2 to: {save_path}")


def plot_diurnal_patterns(df: pd.DataFrame, save_path: Path = FIGURES_DIR / "03_diurnal_hourly_patterns.png"):
    """Visual 3: 24-hour diurnal patterns showing mean and confidence bands."""
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    units = ["°C", "mbar", "%", "m/s"]

    for i, col in enumerate(TARGET_COLS):
        hourly_stats = df_copy.groupby("hour")[col].agg(["mean", "std"])
        hours = hourly_stats.index
        mean = hourly_stats["mean"]
        std = hourly_stats["std"]

        axes[i].plot(hours, mean, color=colors[i], marker="o", linewidth=2, label="Mean Hourly Profile")
        axes[i].fill_between(hours, mean - std, mean + std, color=colors[i], alpha=0.2, label="±1 Std Dev Ribbon")
        axes[i].set_title(f"Diurnal Cycle: {col}", fontweight="bold")
        axes[i].set_xlabel("Hour of Day (0–23)")
        axes[i].set_ylabel(f"{col} ({units[i]})")
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend(loc="best")

    plt.suptitle("Average 24-Hour Diurnal Cycles Across Target Weather Variables", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 3 to: {save_path}")


def plot_cross_variable_relationships(df: pd.DataFrame, save_path: Path = FIGURES_DIR / "04_temp_humidity_pressure_coupling.png"):
    """Visual 4: Cross-variable thermodynamic relationship (Temperature vs Relative Humidity by Season)."""
    df_copy = df.copy()
    # Map months to seasons
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    df_copy["Season"] = df_copy.index.month.map(month_to_season)

    # Subsample 5,000 points evenly across time for a clear scatter plot without clutter
    sample_df = df_copy.sample(n=min(5000, len(df_copy)), random_state=42)

    g = sns.lmplot(
        data=sample_df,
        x="T (degC)",
        y="rh (%)",
        hue="Season",
        palette="viridis",
        aspect=1.4,
        height=6,
        scatter_kws={"alpha": 0.4, "s": 15},
        order=2
    )
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("Thermodynamic Coupling: Temperature vs. Relative Humidity by Season", fontsize=14, fontweight="bold")
    g.set_axis_labels("Temperature (°C)", "Relative Humidity (%)")

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 4 to: {save_path}")


def plot_distributions_and_extremes(df: pd.DataFrame, save_path: Path = FIGURES_DIR / "05_distributions_and_outliers.png"):
    """Visual 5: Target distributions and seasonal boxplots highlighting skewness and extremes."""
    df_copy = df.copy()
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    df_copy["Season"] = df_copy.index.month.map(month_to_season)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    for i, col in enumerate(TARGET_COLS):
        sns.boxplot(
            data=df_copy,
            x="Season",
            y=col,
            order=season_order,
            palette="Set2",
            ax=axes[i],
            fliersize=2,
            boxprops=dict(alpha=0.85)
        )
        axes[i].set_title(f"Distribution by Season: {col}", fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Target Variable Distributions and Extreme Variances Across Seasons", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 5 to: {save_path}")


def run_eda_pipeline(data_path: Path = CLEANED_DATA_PATH):
    """Executes all 5 EDA visualizations on the cleaned hourly dataset."""
    print("\n--- RUNNING EXPLORATORY DATA ANALYSIS PIPELINE ---")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[*] Loaded cleaned dataset from {data_path} with {len(df):,} hourly samples.")

    plot_seasonality_and_trends(df)
    plot_correlation_heatmap(df)
    plot_diurnal_patterns(df)
    plot_cross_variable_relationships(df)
    plot_distributions_and_extremes(df)
    print("[+] All 5 EDA visualizations generated successfully in 'figures/' directory.")


if __name__ == "__main__":
    run_eda_pipeline()
