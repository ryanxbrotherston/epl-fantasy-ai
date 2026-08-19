"""
backtest_risk_analyzer.py — does the "hidden gem" claim actually hold up?

Core question: among players who scored well in a gameweek, did the
LOW-ownership ones actually score MORE than the high-ownership ones on
average (control for price so we're comparing similar-caliber players),
or is "differential" just a name with no real signal behind it?
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"

TEST_SEASON = "2025-26"


def main():
    gw = pd.read_csv(DATA_DIR / TEST_SEASON / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    gw = gw[gw["minutes"] > 0]  # only played players - unused players trivially have 0 ownership signal

    # price band control: compare within similar price tiers so we're not
    # just rediscovering "expensive players score more"
    gw["price_band"] = pd.cut(gw["value"], bins=[0, 50, 70, 90, 999], labels=["budget", "mid", "premium", "elite"])
    gw["ownership_tier"] = pd.cut(gw["selected"].rank(pct=True) * 100, bins=[0, 10, 25, 100],
                                    labels=["low_ownership", "mid_ownership", "high_ownership"])

    summary = gw.groupby(["price_band", "ownership_tier"], observed=True)["total_points"].agg(["mean", "count"])
    print("Average points per gameweek, by price band and ownership tier:")
    print(summary.to_string())

    print("\n--- Within-price-band comparison (low vs high ownership) ---")
    for band in ["budget", "mid", "premium", "elite"]:
        band_data = gw[gw.price_band == band]
        low = band_data[band_data.ownership_tier == "low_ownership"]["total_points"]
        high = band_data[band_data.ownership_tier == "high_ownership"]["total_points"]
        if len(low) > 20 and len(high) > 20:
            print(f"{band}: low-ownership avg={low.mean():.2f} (n={len(low)}), "
                  f"high-ownership avg={high.mean():.2f} (n={len(high)}), "
                  f"diff={low.mean() - high.mean():+.2f}")


if __name__ == "__main__":
    main()
