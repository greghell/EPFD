#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from mwa_pb import primary_beam
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

# ============================================================
# 1) SKY TESSELLATION FUNCTION
# ============================================================

def sky_cells_m1583(niters, step_size=3 * u.deg, lat_range=(0 * u.deg, 90 * u.deg), test=0):
    """
    Tessellate the visible sky following the logic used in ITU-R M.1583.

    Parameters
    ----------
    niters : int
        Number of random pointings per cell. In this script we only need
        the grid definition, so niters can simply be set to 1.
    step_size : astropy quantity
        Approximate cell size in degrees.
    lat_range : tuple
        Elevation range to tessellate, in degrees.
    test : int
        If non-zero, make diagnostic plots.

    Returns
    -------
    tel_az, tel_el : arrays
        Random pointings inside each cell. Not used directly here.
    grid_info : structured array
        Cell definitions, containing:
            cell_lon       = center azimuth
            cell_lat       = center elevation
            cell_lon_low   = lower azimuth edge
            cell_lon_high  = upper azimuth edge
            cell_lat_low   = lower elevation edge
            cell_lat_high  = upper elevation edge
            solid_angle    = approximate cell solid angle proxy
    """

    def sample(niters, low_lon, high_lon, low_lat, high_lat):
        z_low  = np.cos(np.radians(90 - low_lat))
        z_high = np.cos(np.radians(90 - high_lat))

        az = np.random.uniform(low_lon, high_lon, size=niters)
        el = 90 - np.degrees(np.arccos(np.random.uniform(z_low, z_high, size=niters)))
        return az, el

    cell_edges, cell_mids, solid_angles, tel_az, tel_el = [], [], [], [], []

    lat_range = (lat_range[0].to_value(u.deg), lat_range[1].to_value(u.deg))
    ncells_lat = int((lat_range[1] - lat_range[0]) / step_size.to_value(u.deg) + 0.5)

    edge_lats = np.linspace(lat_range[0], lat_range[1], ncells_lat + 1, endpoint=True)
    mid_lats = 0.5 * (edge_lats[1:] + edge_lats[:-1])

    for low_lat, mid_lat, high_lat in zip(edge_lats[:-1], mid_lats, edge_lats[1:]):
        ncells_lon = int(360 * np.cos(np.radians(mid_lat)) / step_size.to_value(u.deg) + 0.5)
        ncells_lon = max(ncells_lon, 1)

        edge_lons = np.linspace(0, 360, ncells_lon + 1, endpoint=True)
        mid_lons = 0.5 * (edge_lons[1:] + edge_lons[:-1])

        solid_angle = (edge_lons[1] - edge_lons[0]) * (
            np.degrees(np.sin(np.radians(high_lat)) - np.sin(np.radians(low_lat)))
        )

        for low_lon, mid_lon, high_lon in zip(edge_lons[:-1], mid_lons, edge_lons[1:]):
            cell_edges.append((low_lon, high_lon, low_lat, high_lat))
            cell_mids.append((mid_lon, mid_lat))
            solid_angles.append(solid_angle)

            cell_tel_az, cell_tel_el = sample(niters, low_lon, high_lon, low_lat, high_lat)
            tel_az.append(cell_tel_az)
            tel_el.append(cell_tel_el)

    tel_az = np.array(tel_az).T
    tel_el = np.array(tel_el).T

    grid_info = np.column_stack([cell_mids, cell_edges, solid_angles])
    grid_info.dtype = np.dtype([
        ("cell_lon", float),
        ("cell_lat", float),
        ("cell_lon_low", float),
        ("cell_lon_high", float),
        ("cell_lat_low", float),
        ("cell_lat_high", float),
        ("solid_angle", float),
    ])

    if test:
        plt.figure(figsize=(10, 6))
        plt.plot(tel_az[0], tel_el[0], "b.")
        plt.xlabel("Azimuth (deg)")
        plt.ylabel("Elevation (deg)")
        plt.grid(True)
        plt.title("Random pointings inside M.1583 sky cells")
        plt.tight_layout()
        plt.show()

    return tel_az, tel_el, grid_info


# ============================================================
# 2) tesselation/beam correction etc.
# ============================================================

def assign_cells_from_grid(df, grid_info):
    """
    Assign each detection to a sky cell using the az/el boundaries from grid_info.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'az' and 'el' in degrees.
    grid_info : structured array
        Output from sky_cells_m1583().

    Returns
    -------
    df : pandas.DataFrame
        Copy of input with an extra integer column 'cell_id'.
        Rows outside the defined grid get cell_id = -1.
    """
    cell_id = np.full(len(df), -1, dtype=int)

    az = df["az"].to_numpy()
    el = df["el"].to_numpy()

    for i, cell in enumerate(grid_info):
        mask = (
            (az >= cell["cell_lon_low"])  &
            (az <  cell["cell_lon_high"]) &
            (el >= cell["cell_lat_low"])  &
            (el <  cell["cell_lat_high"])
        )
        cell_id[mask] = i

    out = df.copy()
    out["cell_id"] = cell_id
    return out


def mwa_analytic_beam_ratio(az_deg, el_deg, freq_hz, pol, delays=None, amps=None, beam_floor=1e-6):
    """
    Compute the normalized analytic beam response at (az, el), divided by the
    zenith response, for the requested polarization.

    This follows your colleague's recommendation:
      correction_factor = beam(az, el) / beam(zenith)

    Parameters
    ----------
    az_deg, el_deg : float
        Azimuth and elevation in degrees.
    freq_hz : float
        Frequency in Hz.
    pol : int
        0 -> XX
        1 -> YY
    delays : array-like or None
        Beamformer delays. Default is all zeros.
    amps : array-like or None
        Dipole amplitudes. Default is one active dipole, matching your
        colleague's example.
    beam_floor : float
        Minimum allowed ratio to avoid division by extremely small numbers.

    Returns
    -------
    ratio : float
        Beam power ratio relative to zenith, for the requested polarization.
    """
    if delays is None:
        delays = np.zeros(16, dtype=int)

    if amps is None:
        amps = np.zeros(16, dtype=int)
        amps[0] = 1

    za_rad = np.radians(90.0 - el_deg)
    az_rad = np.radians(az_deg)

    # Beam at direction of interest
    beam_dir = primary_beam.MWA_Tile_analytic(
        za=za_rad,
        az=az_rad,
        freq=freq_hz,
        delays=delays,
        amps=amps,
        zenithnorm=False,
        power=True,
        jones=False
    )

    # Beam at zenith
    beam_zen = primary_beam.MWA_Tile_analytic(
        za=0.0,
        az=0.0,
        freq=freq_hz,
        delays=delays,
        amps=amps,
        zenithnorm=False,
        power=True,
        jones=False
    )

    beam_dir = np.asarray(beam_dir).squeeze()
    beam_zen = np.asarray(beam_zen).squeeze()

    if beam_dir.ndim == 0:
        ratio = float(beam_dir) / float(beam_zen)
    elif beam_dir.size == 2:
        if pol == 0:
            ratio = float(beam_dir[0]) / float(beam_zen[0])   # XX
        elif pol == 1:
            ratio = float(beam_dir[1]) / float(beam_zen[1])   # YY
        else:
            raise ValueError(f"Unknown polarization index: {pol}")
    else:
        raise ValueError(f"Unexpected beam output shape: {beam_dir.shape}")

    return max(ratio, beam_floor)


def compute_beam_corrected_flux(df, beam_floor=1e-6, delays=None, amps=None):
    """
    Correct measured flux densities using the analytic beam, normalized by
    the zenith response.

    Assumption
    ----------
    The measured catalog fluxes are apparent fluxes attenuated by the beam.
    We divide by:
        beam(az, el) / beam(zenith)
    to estimate a zenith/0-dBi-referenced quantity.

    Returns
    -------
    DataFrame with added columns:
        beam_ratio
        flux_jy_0dBi
        pfd_w_m2_hz_0dBi
        pfd_db_w_m2_hz_0dBi
    """
    beam_ratios = []

    for az_deg, el_deg, freq_hz, pol in zip(df["az"], df["el"], df["freq_hz"], df["pol"]):
        r = mwa_analytic_beam_ratio(
            az_deg=az_deg,
            el_deg=el_deg,
            freq_hz=freq_hz,
            pol=pol,
            delays=delays,
            amps=amps,
            beam_floor=beam_floor
        )
        beam_ratios.append(r)

    out = df.copy()
    out["beam_ratio"] = np.array(beam_ratios)

    # Undo attenuation relative to zenith
    out["flux_jy_0dBi"] = out["flux_jy"] / out["beam_ratio"]

    # Jy -> W/m^2/Hz
    out["pfd_w_m2_hz_0dBi"] = out["flux_jy_0dBi"] * 1e-26
    out["pfd_db_w_m2_hz_0dBi"] = 10 * np.log10(out["pfd_w_m2_hz_0dBi"])

    return out

def add_mwa_sun_elevation(df, time_col="time"):
    """
    Add Sun elevation at the MWA site for each Unix timestamp.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a Unix timestamp column.
    time_col : str
        Name of the timestamp column in seconds since Unix epoch.

    Returns
    -------
    df : pandas.DataFrame
        Copy with added columns:
            sun_el_deg
            is_day
    """
    mwa_location = EarthLocation(
        lat=-26.7033 * u.deg,
        lon=116.6708 * u.deg,
        height=377 * u.m
    )

    times = Time(df[time_col].to_numpy(), format="unix")
    sun_altaz = get_sun(times).transform_to(
        AltAz(obstime=times, location=mwa_location)
    )

    out = df.copy()
    out["sun_el_deg"] = sun_altaz.alt.deg
    out["is_day"] = out["sun_el_deg"] > 0
    return out


# ============================================================
# 3) USER SETTINGS
# ============================================================

CSV_PATH = r"C:\Users\gregh\Desktop\EPFD\identifications_test_v5_starlink_3_min_final_export.csv"

# Select only the stacked image measurements
STACKED_CHANNEL = 31

# Minimum elevation to include in the analysis
MIN_ELEVATION_DEG = 20.0

# Polarization mode:
#   "XX" -> use pol = 0 only
#   "YY" -> use pol = 1 only
#   "I"  -> use both and sum them (Stokes I-like total intensity)
POL_MODE = "XX"

# Restrict to the radio astronomy band
FREQ_MIN_HZ = 145.05e6
FREQ_MAX_HZ = 154.00e6

# Optional time restriction
#TIME_MIN = None   # e.g. 1.72e9
#TIME_MAX = None   # e.g. 1.722e9
TIME_MIN = 1.72e9
TIME_MAX = 1.722e9

# Time binning for "instantaneous" samples
TIME_BIN_SECONDS = 1

# Averaging window length
WINDOW_SECONDS = 2000

# Number of M.1583-like random cells across the sky
CELL_STEP_SIZE = 3 * u.deg

# RA.769 threshold for 150.05–153 MHz continuum
RA769_THRESHOLD_DB_W_M2_HZ = -259.0

# Beam safeguard
BEAM_FLOOR = 1e-6

# Day/night mode:
#   "any"   -> keep all data in the selected time range
#   "day"   -> keep only samples with Sun above horizon
#   "night" -> keep only samples with Sun below horizon
DAY_NIGHT_MODE = "any"

# Analytic beam setup from colleague's example
BEAM_DELAYS = np.zeros(16, dtype=int)
BEAM_AMPS = np.zeros(16, dtype=int)
BEAM_AMPS[0] = 1


# ============================================================
# 4) LOAD THE CATALOG
# ============================================================

cols = [
    "time", "img_idx", "freq_hz", "norad", "fchan",
    "pol", "flux_jy", "range_km",
    "ra_fit", "dec_fit", "ra_tle", "dec_tle",
    "az", "el",
    "x_fit", "y_fit", "x_tle", "y_tle",
    "pix_dist", "x_err", "y_err"
]

df = pd.read_csv(
    CSV_PATH,
    sep=r"\s+",
    names=cols,
    header=None
)

print("Raw shape:", df.shape)


# ============================================================
# 5) FILTER THE DATA
# ============================================================

# Keep only the stacked measurements
df = df[df["fchan"] == STACKED_CHANNEL].copy()

# Frequency selection
df = df[(df["freq_hz"] >= FREQ_MIN_HZ) & (df["freq_hz"] <= FREQ_MAX_HZ)].copy()

# Optional time restriction
if TIME_MIN is not None:
    df = df[df["time"] >= TIME_MIN]
if TIME_MAX is not None:
    df = df[df["time"] <= TIME_MAX]

# Add Sun elevation at MWA for the remaining rows
df = add_mwa_sun_elevation(df, time_col="time")

# Apply optional day/night filter inside the chosen time range
if DAY_NIGHT_MODE == "any":
    pass
elif DAY_NIGHT_MODE == "day":
    df = df[df["is_day"]].copy()
elif DAY_NIGHT_MODE == "night":
    df = df[~df["is_day"]].copy()
else:
    raise ValueError("DAY_NIGHT_MODE must be 'any', 'day', or 'night'")

# Select polarization mode
if POL_MODE == "XX":
    df = df[df["pol"] == 0].copy()
    pol_label = "XX"
elif POL_MODE == "YY":
    df = df[df["pol"] == 1].copy()
    pol_label = "YY"
elif POL_MODE == "I":
    # Keep both linear polarizations; they will be summed later
    df = df[df["pol"].isin([0, 1])].copy()
    pol_label = "Stokes I (XX+YY)"
else:
    raise ValueError("POL_MODE must be 'XX', 'YY', or 'I'")

# Remove bad rows
df = df[np.isfinite(df["flux_jy"])]
df = df[df["flux_jy"] > 0]
df = df[np.isfinite(df["az"]) & np.isfinite(df["el"])]

# Elevation cut
df = df[df["el"] >= MIN_ELEVATION_DEG].copy()

# Convert time to datetime and also keep integer seconds
df["time_dt"] = pd.to_datetime(df["time"], unit="s")
df["time_sec"] = np.floor(df["time"]).astype(int)

print("Shape after filtering:", df.shape)
print("Unique frequencies (MHz):", np.sort(df["freq_hz"].unique() / 1e6))
print(f"Polarization mode: {pol_label}")
print(f"Minimum elevation: {MIN_ELEVATION_DEG:.1f} deg")
print("Rows per polarization after filtering:")
print(df["pol"].value_counts().sort_index())

print(f"Day/night mode: {DAY_NIGHT_MODE}")
if not df.empty:
    print(f"Sun elevation range in filtered data: {df['sun_el_deg'].min():.2f} to {df['sun_el_deg'].max():.2f} deg")
    print("Rows by day/night after filtering:")
    print(df["is_day"].value_counts())

if df.empty:
    raise ValueError("No detections remain after filtering.")


# ============================================================
# 6) BUILD THE SKY GRID
# ============================================================

_, _, grid_info = sky_cells_m1583(
    niters=1,
    step_size=CELL_STEP_SIZE,
    lat_range=(0 * u.deg, 90 * u.deg),
    test=0
)

print("Number of sky cells:", len(grid_info))

# Assign each detection to a cell
df = assign_cells_from_grid(df, grid_info)

# Remove any detections that somehow fall outside the grid
df = df[df["cell_id"] >= 0].copy()

print("Shape after cell assignment:", df.shape)


# ============================================================
# 7) PRIMARY-BEAM CORRECTION TO REFERENCE TO 0 dBi
# ============================================================

df = compute_beam_corrected_flux(
    df,
    beam_floor=BEAM_FLOOR,
    delays=BEAM_DELAYS,
    amps=BEAM_AMPS
)

print("Beam ratio statistics (relative to zenith):")
print(df["beam_ratio"].describe())


# ============================================================
# 8) INSTANTANEOUS EPFD-LIKE SAMPLES PER (TIME, CELL)
# ============================================================

# For each second and each sky cell, sum the 0 dBi-referenced pfd
# contributions of all detections in that cell.
inst = (
    df.groupby(["time_sec", "cell_id"])["pfd_w_m2_hz_0dBi"]
      .sum()
      .reset_index()
)

inst["epfd_db_w_m2_hz_0dBi"] = 10 * np.log10(inst["pfd_w_m2_hz_0dBi"])

print("Number of instantaneous (time,cell) samples:", len(inst))


# ============================================================
# 9) ASSIGN EACH INSTANTANEOUS SAMPLE TO A 2000-s WINDOW
# ============================================================

inst["window"] = (inst["time_sec"] // WINDOW_SECONDS).astype(int)

# Sum detected contributions within each (window, cell)
window_cell_sum = (
    inst.groupby(["window", "cell_id"])["pfd_w_m2_hz_0dBi"]
        .sum()
        .reset_index()
)

# Convert summed power in the window to a time average over the full 2000 s
window_cell_sum["pfd_w_m2_hz_0dBi_avg"] = (
    window_cell_sum["pfd_w_m2_hz_0dBi"] / WINDOW_SECONDS
)

# ============================================================
# 10) EXPAND ONLY TO WINDOW × CELL (NOT TIME × CELL)
# ============================================================

# Keep only windows that actually contain at least one detected contribution
all_windows = np.sort(window_cell_sum["window"].unique())
all_cells = np.arange(len(grid_info))
print(f"Number of active windows kept: {len(all_windows)}")
print(f"Window IDs kept: {all_windows[:10]}{' ...' if len(all_windows) > 10 else ''}")

full_window_index = pd.MultiIndex.from_product(
    [all_windows, all_cells],
    names=["window", "cell_id"]
)

epfd_window = (
    window_cell_sum.set_index(["window", "cell_id"])
                  .reindex(full_window_index)
                  .reset_index()
)

# Cells with no detections in a window are assigned zero contribution
epfd_window["pfd_w_m2_hz_0dBi_avg"] = epfd_window["pfd_w_m2_hz_0dBi_avg"].fillna(0.0)

# Convert to dB only for nonzero values
epfd_window["epfd_db_w_m2_hz_0dBi"] = np.where(
    epfd_window["pfd_w_m2_hz_0dBi_avg"] > 0,
    10 * np.log10(epfd_window["pfd_w_m2_hz_0dBi_avg"]),
    np.nan
)

n_total = len(epfd_window)
n_nonzero = np.sum(epfd_window["pfd_w_m2_hz_0dBi_avg"] > 0)

print(f"Total active window×cell samples: {n_total}")
print(f"Nonzero active window×cell samples: {n_nonzero}")
print(f"Fraction nonzero: {100*n_nonzero/n_total:.6f} %")

# Keep linear values for threshold comparison
linear_values = epfd_window["pfd_w_m2_hz_0dBi_avg"].to_numpy()

# Keep dB values for plotting
values_db = epfd_window["epfd_db_w_m2_hz_0dBi"].dropna().to_numpy()

print("Number of averaged window×cell samples:", len(linear_values))
print("Number of nonzero samples for plotting:", len(values_db))

# ============================================================
# 11) THRESHOLD COMPARISON
# ============================================================

threshold_linear = 10 ** (RA769_THRESHOLD_DB_W_M2_HZ / 10.0)

exceedance = np.mean(linear_values > threshold_linear)

print(f"RA.769 threshold: {RA769_THRESHOLD_DB_W_M2_HZ:.1f} dB(W/m^2/Hz)")
print(f"Fraction above threshold: {100 * exceedance:.6f} %")

# ============================================================
# 12) CDF
# ============================================================

sorted_vals = np.sort(values_db)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

# Histogram
plt.figure(figsize=(10, 6))
counts, bins = np.histogram(values_db, bins=100)
counts_percent = counts / counts.sum() * 100

plt.bar(
    bins[:-1],
    counts_percent,
    width=np.diff(bins),
    align="edge",
    alpha=0.6,
    label="Nonzero EPFD samples"
)

plt.axvline(
    RA769_THRESHOLD_DB_W_M2_HZ,
    linestyle="--",
    linewidth=2,
    label=f"RA.769 threshold = {RA769_THRESHOLD_DB_W_M2_HZ:.1f} dB(W/m²/Hz)"
)

plt.xlabel("EPFD proxy referenced to 0 dBi [dB(W/m²/Hz)]")
plt.ylabel("Percentage of nonzero samples (%)")
plt.title("EPFD-like distribution in the 150.05–153 MHz band")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# CDF
plt.figure(figsize=(10, 6))
plt.plot(sorted_vals, cdf, drawstyle="steps-post", linewidth=2, label="Empirical CDF")

plt.axvline(
    RA769_THRESHOLD_DB_W_M2_HZ,
    linestyle="--",
    linewidth=2,
    label=f"RA.769 threshold = {RA769_THRESHOLD_DB_W_M2_HZ:.1f} dB(W/m²/Hz)"
)

plt.xlabel("EPFD proxy referenced to 0 dBi [dB(W/m²/Hz)]")
plt.ylabel("Cumulative probability")
plt.title("EPFD-like CDF in the 150.05–153 MHz band")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# CDF INCLUDING ZEROS, BUT DISPLAYED IN dB
# ============================================================

linear_values = epfd_window["pfd_w_m2_hz_0dBi_avg"].to_numpy()

# Sort ALL values (including zeros)
sorted_linear = np.sort(linear_values)

# Build CDF (0 → 1)
cdf = np.arange(1, len(sorted_linear) + 1) / len(sorted_linear)

# Convert to dB for plotting
# Avoid log(0) by masking zeros
sorted_db = np.full_like(sorted_linear, np.nan, dtype=float)
mask_nonzero = sorted_linear > 0
sorted_db[mask_nonzero] = 10 * np.log10(sorted_linear[mask_nonzero])

# Keep only finite values for plotting
valid = np.isfinite(sorted_db)
sorted_db_plot = sorted_db[valid]
cdf_plot = cdf[valid]

# Convert y-axis to %
cdf_plot_percent = cdf_plot * 100

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(sorted_db_plot, cdf_plot_percent, drawstyle="steps-post", linewidth=2, label="Empirical CDF (all samples)")

plt.axvline(
    RA769_THRESHOLD_DB_W_M2_HZ,
    linestyle="--",
    linewidth=2,
    label=f"RA.769 threshold = {RA769_THRESHOLD_DB_W_M2_HZ:.1f} dB(W/m²/Hz)"
)

zero_fraction = 100 * np.mean(linear_values == 0)

plt.text(
    sorted_db_plot[0],
    cdf_plot_percent[0] + 2,
    f"{zero_fraction:.1f}% of samples = 0",
    fontsize=10
)

plt.xlabel("EPFD proxy referenced to 0 dBi [dB(W/m²/Hz)]")
plt.ylabel("Cumulative probability [%]")
plt.title("EPFD CDF")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


'''
MAKE GIF
'''
'''
import os
import matplotlib.pyplot as plt
import imageio
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

cell_az = np.array([cell["cell_lon"] for cell in grid_info])
cell_el = np.array([cell["cell_lat"] for cell in grid_info])
cell_r = np.radians(90 - cell_el)
cell_theta = np.radians(cell_az)

unique_windows = np.sort(epfd_window["window"].unique())

filenames = []

vmin = np.nanpercentile(epfd_window["epfd_db_w_m2_hz_0dBi"], 5)
vmax = np.nanpercentile(epfd_window["epfd_db_w_m2_hz_0dBi"], 95)

for w in unique_windows:
    df_w = epfd_window[epfd_window["window"] == w]

    # Nonzero cells only
    mask = df_w["pfd_w_m2_hz_0dBi_avg"] > 0

    # Values for coloring (in dB)
    values_db = df_w.loc[mask, "epfd_db_w_m2_hz_0dBi"].to_numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')

    ax.set_title(f"Window {w}", fontsize=12)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)

    # Plot all cells as faint background
    ax.scatter(cell_theta, cell_r, s=2, alpha=0.05)

    # Plot active cells
    if np.any(mask):
        sc = ax.scatter(
            cell_theta[df_w["cell_id"][mask]],
            cell_r[df_w["cell_id"][mask]],
            c=values_db,
            s=10,
            vmin=vmin,
            vmax=vmax,
        )

        plt.colorbar(sc, ax=ax, label="EPFD [dB(W/m²/Hz)]")

    fname = os.path.join(frames_dir, f"frame_{w:04d}.png")
    plt.savefig(fname, dpi=120)
    plt.close()

    filenames.append(fname)

gif_path = "epfd_sky_evolution.gif"

with imageio.get_writer(gif_path, mode="I", duration=0.5) as writer:
    for fname in filenames:
        image = imageio.imread(fname)
        writer.append_data(image)

print(f"GIF saved to {gif_path}")
'''










































