"""
Tangara Timetable Simulation v2.0 (Python port)
Simulates progressive train service reductions and passenger redistribution.
"""

import re
import numpy as np
import pandas as pd

# --- Configuration -----------------------------------------------------------
# List all input CSV files — one per train line. Each line is processed independently.
INPUT_FILES = [
    "Timetable Sheet_Full Data_down.csv",
    # "Line2_data.csv",
    # "Line3_data.csv",
]
# Fixed 6-trains-per-hour scenario: both bounds set to 6 so only one iteration runs.
SERVICE_CAP_START = 6
SERVICE_CAP_END = 6
SERVICE_CAP_STEP = 2
TRAINS_TO_CANCEL_PER_HOUR = 2
TRAIN_CAPACITY = 840
REDISTRIBUTION_RATE = 0.7   # Improvement 3: was hardcoded 0.7/0.3 throughout R code
CANCELLATION_RATE = 1 - REDISTRIBUTION_RATE

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
LOAD_BREAKS = list(range(0, 201, 10)) + [float("inf")]
LOAD_LABELS = [f"{lo}-{hi}" for lo, hi in zip(range(0, 200, 10), range(10, 201, 10))] + ["200+"]


# --- Data loading & cleaning -------------------------------------------------

def decimal_to_hhmm(val: str) -> str:
    """Convert a decimal hour string (e.g. '8.5') to 'HH:MM'."""
    hours, frac = divmod(float(val), 1)
    minutes = round(frac * 60)   # Improvement 1: fixes '*a' typo from R source
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{int(hours):02d}:{int(minutes):02d}"


def fix_hhmm_60(val: str) -> str:
    """Carry over XX:60 to next hour."""
    h, m = val.split(":")
    if m == "60":
        return f"{int(h) + 1:02d}:00"
    return f"{int(h):02d}:{m}"


def load_and_clean(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, header=0)

    # Identify columns that contain at least some HH:MM values
    time_cols = [
        col for col in data.columns
        if data[col].astype(str).str.match(r"\d{1,2}:\d{2}").any()
    ]

    for col in time_cols:
        col_str = data[col].astype(str)
        is_decimal = ~col_str.str.match(r"\d{1,2}:\d{2}")
        data.loc[is_decimal, col] = col_str[is_decimal].apply(decimal_to_hhmm)
        data[col] = data[col].astype(str).apply(fix_hhmm_60)

    # Pad and parse departure time
    data["Timetabled.Departure.Time"] = (
        data["Timetabled.Departure.Time"]
        .astype(str)
        .str.strip()
        .apply(lambda t: t if len(t.split(":")[0]) == 2 else "0" + t)
    )
    data["Timetabled.Departure.Time"] = pd.to_datetime(
        data["Timetabled.Departure.Time"], format="%H:%M"
    )

    data.sort_values(["Station.Name", "Timetabled.Departure.Time"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


# --- Passenger redistribution ------------------------------------------------

def redistribute_passengers(
    data: pd.DataFrame,
    removed_train,
    redistribution_rate: float = REDISTRIBUTION_RATE,
) -> dict:
    """
    Cancel `removed_train`, redistribute passengers to the next available
    service at each station, and return updated data + metrics.

    Improvement 2: next-train lookup uses vectorized pandas masking instead of
    a Python-level row loop, reducing O(n*stations) overhead.
    """
    cancellation_rate = 1 - redistribution_rate
    removed_mask = data["Joined.Run.Number"] == removed_train
    removed_data = data[removed_mask].copy()

    if removed_data.empty:
        print(f"No data found for train: {removed_train}. Skipping.")
        return {"data": data, "redistributed": 0, "cancelled": 0, "avg_delay": np.nan}

    redistributed_total = 0.0
    cancelled_total = 0.0
    delay_times = []

    for _, row in removed_data.iterrows():
        station = row["Station.Name"]
        dep_time = row["Timetabled.Departure.Time"]
        board = row["Board"]

        # Vectorized mask for next available trains at this station
        next_mask = (
            (data["Station.Name"] == station)
            & (data["Timetabled.Departure.Time"] > dep_time)
            & (data["Joined.Run.Number"] != removed_train)
        )
        next_data = data[next_mask]

        if next_data.empty:
            cancelled_total += board
            continue

        closest_time = next_data["Timetabled.Departure.Time"].min()
        closest_mask = next_mask & (data["Timetabled.Departure.Time"] == closest_time)
        n_next = closest_mask.sum()

        to_redistribute = redistribution_rate * board
        to_cancel = cancellation_rate * board

        # Boardings distributed evenly across all trains at that time
        data.loc[closest_mask, "Board"] += to_redistribute / n_next
        redistributed_total += to_redistribute
        cancelled_total += to_cancel

        # Alightings adjusted proportionally to existing alight pattern
        alight_sum = data.loc[closest_mask, "Alight"].sum()
        if alight_sum > 0:
            alight_pattern = data.loc[closest_mask, "Alight"] / alight_sum
        else:
            alight_pattern = pd.Series([1 / n_next] * n_next, index=data.loc[closest_mask].index)
        data.loc[closest_mask, "Alight"] += to_redistribute * alight_pattern

        # Recalculate departure load (vectorized)
        data.loc[closest_mask, "Departure.Load"] = (
            data.loc[closest_mask, "Board"]
            - data.loc[closest_mask, "Alight"]
            + data.loc[closest_mask, "Departure.Load"]
        )

        delay_mins = (closest_time - dep_time).total_seconds() / 60
        delay_times.extend([delay_mins] * int(round(to_redistribute)))

    # Drop cancelled train
    data = data[~removed_mask].copy()
    data["Adjusted.Departure.Load.Percentage"] = data["Departure.Load"] / TRAIN_CAPACITY * 100

    avg_delay = float(np.nanmean(delay_times)) if delay_times else np.nan
    return {
        "data": data,
        "redistributed": redistributed_total,
        "cancelled": cancelled_total,
        "avg_delay": avg_delay,
    }


# --- Main simulation loop ----------------------------------------------------

def run_simulation(data: pd.DataFrame, output_prefix: str = ""):
    saved_data = {}
    summary_tables = []
    redistribution_summary_rows = []

    service_cap = SERVICE_CAP_START

    while service_cap >= SERVICE_CAP_END:
        label = f"service_cap_{service_cap}"
        print(f"\n=== Running simulation for {label} ===")

        # Count trains per hour; keep only hours with >= service_cap trains
        trains_per_hour = data.groupby("Central.Hour")["Joined.Run.Number"].nunique()
        eligible_hours = trains_per_hour[trains_per_hour >= service_cap].index

        avg_departure_loads = (
            data[data["Central.Hour"].isin(eligible_hours)]
            .groupby(["Central.Hour", "Joined.Run.Number"])["Departure.Load"]
            .mean()
            .reset_index()
            .rename(columns={"Departure.Load": "avg_departure_load"})
            .sort_values(["Central.Hour", "avg_departure_load"])
        )

        # Select the 2 least-loaded trains per eligible hour
        trains_to_cancel = (
            avg_departure_loads
            .groupby("Central.Hour")
            .head(TRAINS_TO_CANCEL_PER_HOUR)["Joined.Run.Number"]
            .tolist()
        )

        redistributed_counts = {}
        cancelled_counts = {}
        avg_delays = []

        for train in trains_to_cancel:
            result = redistribute_passengers(data, train)
            data = result["data"]
            redistributed_counts[train] = result["redistributed"]
            cancelled_counts[train] = result["cancelled"]
            avg_delays.append(result["avg_delay"])
            print(
                f"  Train {train}: redistributed={result['redistributed']:.1f}, "
                f"cancelled={result['cancelled']:.1f}, delay={result['avg_delay']:.1f} min"
            )

        # Categorise load
        data["Adjusted.Departure.Load.Category"] = pd.cut(
            data["Adjusted.Departure.Load.Percentage"],
            bins=LOAD_BREAKS,
            labels=LOAD_LABELS,
            right=False,
            include_lowest=True,
        )

        # Iteration summary
        iter_summary = (
            data.groupby("Adjusted.Departure.Load.Category", observed=True)["Departure.Load"]
            .sum()
            .reset_index()
            .rename(columns={"Departure.Load": "Total_Departure_Load"})
        )
        iter_summary.insert(0, "Iteration", label)
        summary_tables.append(iter_summary)

        redistribution_summary_rows.append({
            "service_cap": service_cap,
            "total_redistributed": sum(redistributed_counts.values()),
            "total_cancelled": sum(cancelled_counts.values()),
            "average_delay": float(np.nanmean(avg_delays)),
        })

        saved_data[label] = data.copy()
        data.to_csv(f"{output_prefix}ProcessedData_{label}.csv", index=False)

        service_cap -= SERVICE_CAP_STEP

    return saved_data, summary_tables, redistribution_summary_rows


# --- Post-loop aggregation & exports -----------------------------------------

def export_results(saved_data, summary_tables, redistribution_summary_rows, output_prefix: str = ""):
    pd.concat(summary_tables, ignore_index=True).to_csv(f"{output_prefix}SummaryTable.csv", index=False)
    pd.DataFrame(redistribution_summary_rows).to_csv(f"{output_prefix}RedistributionSummary.csv", index=False)

    # Boardings per station per service cap (wide format)
    station_frames = []
    for cap_label, df in saved_data.items():
        agg = df.groupby("Station.Name")["Board"].sum().reset_index()
        agg["service_cap"] = cap_label
        station_frames.append(agg)

    combined = pd.concat(station_frames, ignore_index=True)
    wide = combined.pivot_table(
        index="Station.Name", columns="service_cap", values="Board", aggfunc="sum"
    ).fillna(0)
    wide.to_csv(f"{output_prefix}Boardings_by_station.csv")
    print(f"\nExports complete{f' for {output_prefix.rstrip(\"_\")}' if output_prefix else ''}.")


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    import os

    for filepath in INPUT_FILES:
        line_name = os.path.splitext(os.path.basename(filepath))[0].replace(" ", "_")
        prefix = f"{line_name}_"
        print(f"\n{'=' * 60}")
        print(f"Processing line: {line_name}  (file: {filepath})")
        print(f"{'=' * 60}")
        data = load_and_clean(filepath)
        saved_data, summary_tables, redistribution_summary_rows = run_simulation(data, output_prefix=prefix)
        export_results(saved_data, summary_tables, redistribution_summary_rows, output_prefix=prefix)
