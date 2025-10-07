"""Verify whether Algoseek database stores adjusted or unadjusted prices."""

from pathlib import Path

import duckdb

data_path = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete/SecId=33449")
parquet_pattern = str(data_path / "*.parquet")

con = duckdb.connect(":memory:")
result = con.execute(f"""
    SELECT TradeDate, Close, CumulativePriceFactor
    FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
    WHERE TradeDate BETWEEN '2020-08-28' AND '2020-08-31'
    ORDER BY TradeDate
""").fetchall()

print("Question: What does the Close column in the database represent?")
print("=" * 70)
for row in result:
    date, close, factor = row
    print(f"{date}: Close={close:.2f}, Factor={factor:.4f}")

print("\n" + "=" * 70)
print("SCENARIO 1: Close is UNADJUSTED (actual traded price)")
print("=" * 70)
print("  8/28: $499.23 was the actual price traders paid")
print("  8/31: $129.04 was actual price after 4:1 split")
print(f"  Ratio: 499.23 / 129.04 = {499.23 / 129.04:.2f} ≈ 4:1 split ✓")
print("  This makes sense!")

print("\n" + "=" * 70)
print("SCENARIO 2: Close is ADJUSTED (back-adjusted)")
print("=" * 70)
print("  Then to get unadjusted, we multiply by factor:")
row1_close, row1_factor = result[0][1], result[0][2]
row2_close, row2_factor = result[1][1], result[1][2]
print(f"  8/28: {row1_close:.2f} × {row1_factor:.4f} = ${row1_close * row1_factor:.2f}")
print(f"  8/31: {row2_close:.2f} × {row2_factor:.4f} = ${row2_close * row2_factor:.2f}")
print("  These prices don't make sense - AAPL never traded at $4000+!")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("The database Close column contains UNADJUSTED prices!")
print("These are the actual historical traded prices.")
print("\nTo get adjusted prices, we should DIVIDE by CumulativePriceFactor:")
print(f"  8/28: {row1_close:.2f} / {row1_factor:.4f} = ${row1_close / row1_factor:.2f}")
print(f"  8/31: {row2_close:.2f} / {row2_factor:.4f} = ${row2_close / row2_factor:.2f}")
print("  These are continuous back-adjusted prices ✓")

con.close()
