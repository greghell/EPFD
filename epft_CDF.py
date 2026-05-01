#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from mwa_pb import primary_beam

C = 299792458.0

CATALOG_CSV = r"C:\Users\gregh\Desktop\EPFD\identifications_test_v5_starlink_3_min_final_export.csv" # Grigg et al. detection dataset
COORDS_CSV  = r"C:\Users\gregh\Desktop\EPFD\coords_xy.csv"  # EDA2 antennas coordinates

POL_MODE = "XX"      # "XX" or "YY"
FREQ_HZ = 150.0e6

# ensures the detections all fall in the 150.05-153 MHz RAS protected band
FREQ_MIN_HZ = 145.05e6
FREQ_MAX_HZ = 154.00e6

# all time samples
TIME_MIN = 1720923916
TIME_MAX = 1721617384

STACKED_CHANNEL = 31            # full 0.9 MHz channels
MIN_ELEVATION_DEG = 20.0        # discard detections below 20 degrees in elevation

SNAPSHOT_SECONDS = 2            # integration of each frame in the data set
WINDOW_SAMPLES = 1000           # 1000 snapshots × 2 s = 2000 s
OVERLAP_PERCENT = 0.0           # overlap in % between windows - 0% = no overlap

CELL_STEP_SIZE = 3 * u.deg
RA769_THRESHOLD_DB_W_M2_HZ = -259.0

N_ANT = 256

# Choose gain convention:
# "N"  -> kernel = N * normalized_synthesis * primary_gain
# "N2" -> kernel = N^2 * normalized_synthesis * primary_gain --> NOT APPLICABLE
ARRAY_GAIN_MODE = "N"

DAY_NIGHT_MODE = "night"         # "any", "day", "night"

BEAM_DELAYS = np.zeros(16, dtype=int)
BEAM_AMPS = np.zeros(16, dtype=int)
BEAM_AMPS[0] = 1


def sky_cells_m1583(niters, step_size=3 * u.deg, lat_range=(0 * u.deg, 90 * u.deg), test=0):

    def sample(niters, low_lon, high_lon, low_lat, high_lat):
        z_low = np.cos(np.radians(90 - low_lat))
        z_high = np.cos(np.radians(90 - high_lat))
        az = np.random.uniform(low_lon, high_lon, size=niters)
        el = 90 - np.degrees(np.arccos(np.random.uniform(z_low, z_high, size=niters)))
        return az, el

    cell_edges, cell_mids, solid_angles, tel_az, tel_el = [], [], [], [], []

    lat_range = (lat_range[0].to_value(u.deg), lat_range[1].to_value(u.deg))
    ncells_lat = int((lat_range[1] - lat_range[0]) / step_size.to_value(u.deg) + 0.5)

    edge_lats = np.linspace(lat_range[0], lat_range[1], ncells_lat + 1)
    mid_lats = 0.5 * (edge_lats[1:] + edge_lats[:-1])

    for low_lat, mid_lat, high_lat in zip(edge_lats[:-1], mid_lats, edge_lats[1:]):
        ncells_lon = int(360 * np.cos(np.radians(mid_lat)) / step_size.to_value(u.deg) + 0.5)
        ncells_lon = max(ncells_lon, 1)

        edge_lons = np.linspace(0, 360, ncells_lon + 1)
        mid_lons = 0.5 * (edge_lons[1:] + edge_lons[:-1])

        solid_angle = (edge_lons[1] - edge_lons[0]) * (
            np.degrees(np.sin(np.radians(high_lat)) - np.sin(np.radians(low_lat)))
        )

        for low_lon, mid_lon, high_lon in zip(edge_lons[:-1], mid_lons, edge_lons[1:]):
            cell_edges.append((low_lon, high_lon, low_lat, high_lat))
            cell_mids.append((mid_lon, mid_lat))
            solid_angles.append(solid_angle)

            a, e = sample(niters, low_lon, high_lon, low_lat, high_lat)
            tel_az.append(a)
            tel_el.append(e)

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

    return tel_az, tel_el, grid_info


def assign_cells_from_grid(df, grid_info):
    cell_id = np.full(len(df), -1, dtype=int)
    az = df["az"].to_numpy()
    el = df["el"].to_numpy()

    for i, cell in enumerate(grid_info):
        mask = (
            (az >= cell["cell_lon_low"]) &
            (az <  cell["cell_lon_high"]) &
            (el >= cell["cell_lat_low"]) &
            (el <  cell["cell_lat_high"])
        )
        cell_id[mask] = i

    out = df.copy()
    out["cell_id"] = cell_id
    return out


def load_array_coordinates(csv_path):
    df = pd.read_csv(csv_path)
    x = df["x_centered"].to_numpy(dtype=float)
    y = df["y_centered"].to_numpy(dtype=float)
    z = np.zeros_like(x)
    return np.column_stack([x, y, z])


def azel_to_unit_vector(az_deg, el_deg):
    az = np.radians(az_deg)
    el = np.radians(el_deg)

    east = np.cos(el) * np.sin(az)
    north = np.cos(el) * np.cos(az)
    up = np.sin(el)

    return np.stack([east, north, up], axis=-1)


def array_factor_for_pointing(
    xyz_m,
    freq_hz,
    point_az_deg,
    point_el_deg,
    sample_az_deg,
    sample_el_deg,
    weights=None,
):
    """
    Normalized synthesis beam power pattern.
    Peak is approximately 1 at the pointing direction.
    Absolute array gain is applied later through ARRAY_GAIN_MODE.
    """
    xyz_m = np.asarray(xyz_m, dtype=float)
    n_elem = xyz_m.shape[0]

    if weights is None:
        weights = np.ones(n_elem, dtype=complex)
    else:
        weights = np.asarray(weights, dtype=complex)
        if len(weights) != n_elem:
            raise ValueError("weights must have same length as number of elements")

    sample_az_deg = np.asarray(sample_az_deg, dtype=float).reshape(-1)
    sample_el_deg = np.asarray(sample_el_deg, dtype=float).reshape(-1)

    lam = C / freq_hz
    k = 2.0 * np.pi / lam

    s0 = np.asarray(
        azel_to_unit_vector(point_az_deg, point_el_deg),
        dtype=float
    ).reshape(3)

    s = np.asarray(
        azel_to_unit_vector(sample_az_deg, sample_el_deg),
        dtype=float
    ).reshape(-1, 3)

    ds = s - s0
    phase = k * (ds @ xyz_m.T)  # shape: n_dir × n_ant

    field = np.sum(weights[None, :] * np.exp(1j * phase), axis=1)
    field_ref = np.sum(weights)

    power = np.abs(field / field_ref) ** 2
    return np.asarray(power, dtype=float).reshape(-1)


def mwa_primary_beam_power_analytic(
    az_deg,
    el_deg,
    freq_hz=150e6,
    pol="XX",
    delays=None,
    amps=None,
):
    if delays is None:
        delays = np.zeros(16, dtype=int)

    if amps is None:
        amps = np.zeros(16, dtype=int)
        amps[0] = 1

    za_rad = np.radians(90.0 - el_deg)
    az_rad = np.radians(az_deg)

    beam_dir = primary_beam.MWA_Tile_analytic(
        za=za_rad,
        az=az_rad,
        freq=freq_hz,
        delays=delays,
        amps=amps,
        zenithnorm=False,
        power=True,
        jones=False,
    )

    beam_dir = np.asarray(beam_dir).squeeze()

    if beam_dir.size == 1:
        return float(beam_dir)

    if pol.upper() == "XX":
        return float(beam_dir[0])
    elif pol.upper() == "YY":
        return float(beam_dir[1])
    else:
        raise ValueError("pol must be 'XX' or 'YY'")


PRIMARY_GAIN_METHOD = "tessellation"    # "tessellation" (same as EPFD analysis) or "grid_proxy" (regular grid in Az/El)
GRID_PROXY_WEIGHTED = False             # False = pixel count proxy
                                        # True  = sin(ZA)-weighted integral
DIPOLE_INDEX = 6                        # arbitrary, 0-255


def compute_primary_beam_raw_on_tessellation(
    grid_info,
    freq_hz=150e6,
    pol="XX",
    delays=None,
    amps=None,
    primary_floor=1e-30,
):
    """
    Evaluate raw MWA analytic primary beam on the tessellation cell centers.
    """
    if delays is None:
        delays = np.zeros(16, dtype=int)

    if amps is None:
        amps = np.zeros(16, dtype=int)
        amps[DIPOLE_INDEX] = 1

    cell_az = np.asarray(grid_info["cell_lon"], dtype=float).reshape(-1)
    cell_el = np.asarray(grid_info["cell_lat"], dtype=float).reshape(-1)

    primary_raw = np.array([
        mwa_primary_beam_power_analytic(
            az_deg=a,
            el_deg=e,
            freq_hz=freq_hz,
            pol=pol,
            delays=delays,
            amps=amps,
        )
        for a, e in zip(cell_az, cell_el)
    ], dtype=float)

    primary_raw = np.maximum(primary_raw, primary_floor)

    return cell_az, cell_el, primary_raw


def compute_primary_gain_tessellation(
    grid_info,
    freq_hz=150e6,
    pol="XX",
    delays=None,
    amps=None,
    primary_floor=1e-30,
):
    """
    Convert raw primary beam to dBi using the M.1583 sky-cell solid-angle integral.
    G(az,el) = 4*pi*B(az,el) / integral_visible[B dOmega] (approx. 0 dBi)
    """
    cell_az, cell_el, primary_raw = compute_primary_beam_raw_on_tessellation(
        grid_info=grid_info,
        freq_hz=freq_hz,
        pol=pol,
        delays=delays,
        amps=amps,
        primary_floor=primary_floor,
    )

    az_low  = np.radians(np.asarray(grid_info["cell_lon_low"], dtype=float).reshape(-1))
    az_high = np.radians(np.asarray(grid_info["cell_lon_high"], dtype=float).reshape(-1))
    el_low  = np.radians(np.asarray(grid_info["cell_lat_low"], dtype=float).reshape(-1))
    el_high = np.radians(np.asarray(grid_info["cell_lat_high"], dtype=float).reshape(-1))

    # Exact solid angle of az/el cell:
    # dOmega = dAz * d(sin El)
    solid_angle_sr = (az_high - az_low) * (np.sin(el_high) - np.sin(el_low))

    beam_integral = np.sum(primary_raw * solid_angle_sr)

    primary_gain_linear = 4.0 * np.pi * primary_raw / beam_integral
    primary_gain_dbi = 10.0 * np.log10(primary_gain_linear)

    return pd.DataFrame({
        "cell_id": np.arange(len(cell_az)),
        "az_deg": cell_az,
        "el_deg": cell_el,
        "solid_angle_sr": solid_angle_sr,
        "primary_raw": primary_raw,
        "primary_gain_linear": primary_gain_linear,
        "primary_gain_dbi": primary_gain_dbi,
        "method": "tessellation",
        "beam_integral": beam_integral,
    })


def estimate_primary_gain_grid_proxy(
    freq_hz=150e6,
    pol="XX",
    delays=None,
    amps=None,
    dipole_index=6,
    weighted=False,
    n_za=91,
    n_az=361,
):
    """
    Evaluate MWA analytic primary beam on regular grid in Az/El
    If weighted=False:
        uses pixel-count proxy:
            gain = peak * (2*Npix) / sum(beam)
    If weighted=True:
        uses proper sin(ZA) solid-angle weighting:
            gain = 4*pi*peak / integral_visible[B dOmega]
    """
    if delays is None:
        delays = np.zeros(16, dtype=int)

    if amps is None:
        amps = np.zeros(16, dtype=int)
        amps[dipole_index] = 1

    za = np.linspace(0.0, np.pi / 2.0, n_za)
    az = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)

    za_grid, az_grid = np.meshgrid(za, az, indexing="ij")

    beam = primary_beam.MWA_Tile_analytic(
        za=za_grid,
        az=az_grid,
        freq=freq_hz,
        delays=delays,
        amps=amps,
        zenithnorm=False,
        power=True,
        jones=False,
    )

    beam = np.asarray(beam).squeeze()

    if beam.ndim == 3:
        if pol.upper() == "XX":
            b = beam[0]
        elif pol.upper() == "YY":
            b = beam[1]
        else:
            raise ValueError("pol must be 'XX' or 'YY'")
    else:
        b = beam

    peak_power = np.nanmax(b)

    if weighted:
        dza = za[1] - za[0]
        daz = az[1] - az[0]
        domega = np.sin(za_grid) * dza * daz
        beam_integral = np.nansum(b * domega)
        peak_gain_linear = 4.0 * np.pi * peak_power / beam_integral
        method = "grid_proxy_weighted"
    else:
        total_power_proxy = np.nansum(b)
        size_proxy = 2.0 * b.size
        peak_gain_linear = peak_power * size_proxy / total_power_proxy
        beam_integral = total_power_proxy
        method = "grid_proxy_unweighted"

    peak_gain_dbi = 10.0 * np.log10(peak_gain_linear)

    return {
        "method": method,
        "peak_power_raw": peak_power,
        "beam_integral_or_sum": beam_integral,
        "peak_gain_linear": peak_gain_linear,
        "peak_gain_dbi": peak_gain_dbi,
    }


def compute_primary_gain_grid_proxy_on_tessellation(
    grid_info,
    freq_hz=150e6,
    pol="XX",
    delays=None,
    amps=None,
    primary_floor=1e-30,
    dipole_index=6,
    weighted=False,
    n_za=91,
    n_az=361,
):
    """
    Peak normalization, then scale the tessellation beam
    so that its maximum equals the estimated peak gain.
    """
    cell_az, cell_el, primary_raw = compute_primary_beam_raw_on_tessellation(
        grid_info=grid_info,
        freq_hz=freq_hz,
        pol=pol,
        delays=delays,
        amps=amps,
        primary_floor=primary_floor,
    )

    proxy = estimate_primary_gain_grid_proxy(
        freq_hz=freq_hz,
        pol=pol,
        delays=delays,
        amps=amps,
        dipole_index=dipole_index,
        weighted=weighted,
        n_za=n_za,
        n_az=n_az,
    )

    raw_peak = np.nanmax(primary_raw)

    primary_gain_linear = primary_raw / raw_peak * proxy["peak_gain_linear"]
    primary_gain_dbi = 10.0 * np.log10(primary_gain_linear)

    return pd.DataFrame({
        "cell_id": np.arange(len(cell_az)),
        "az_deg": cell_az,
        "el_deg": cell_el,
        "primary_raw": primary_raw,
        "primary_gain_linear": primary_gain_linear,
        "primary_gain_dbi": primary_gain_dbi,
        "method": proxy["method"],
        "proxy_peak_gain_linear": proxy["peak_gain_linear"],
        "proxy_peak_gain_dbi": proxy["peak_gain_dbi"],
        "proxy_peak_power_raw": proxy["peak_power_raw"],
        "proxy_beam_integral_or_sum": proxy["beam_integral_or_sum"],
    })


def compute_primary_gain_on_tessellation(
    grid_info,
    freq_hz=150e6,
    pol="XX",
    method="tessellation",
    weighted_grid_proxy=False,
    delays=None,
    amps=None,
    dipole_index=6,
):
    """
    Wrapper allowing comparison between:
        method="tessellation"  -> solid-angle integral over sky cells
        method="grid_proxy"    -> grid proxy
    (test only)
    """
    method = method.lower()

    if method == "tessellation":
        df = compute_primary_gain_tessellation(
            grid_info=grid_info,
            freq_hz=freq_hz,
            pol=pol,
            delays=delays,
            amps=amps,
        )

    elif method == "grid_proxy":
        df = compute_primary_gain_grid_proxy_on_tessellation(
            grid_info=grid_info,
            freq_hz=freq_hz,
            pol=pol,
            delays=delays,
            amps=amps,
            dipole_index=dipole_index,
            weighted=weighted_grid_proxy,
        )

    else:
        raise ValueError("method must be 'tessellation' or 'grid_proxy'")

    print("\nPrimary beam gain normalization")
    print("--------------------------------")
    print(f"Method: {df['method'].iloc[0]}")
    print(f"Max gain: {df['primary_gain_linear'].max():.4f} linear")
    print(f"Max gain: {df['primary_gain_dbi'].max():.2f} dBi")
    print(f"Median gain: {df['primary_gain_dbi'].median():.2f} dBi")
    print(f"Min gain: {df['primary_gain_dbi'].min():.2f} dBi")

    return df


def get_array_gain_factor(n_ant, mode="N"):
    if mode.upper() == "N":
        return float(n_ant)
    elif mode.upper() == "N2":
        return float(n_ant ** 2)
    else:
        raise ValueError("ARRAY_GAIN_MODE must be 'N' or 'N2'")


def compute_convolution_kernel_for_source_cell(
    source_cell_id,
    xyz_m,
    grid_info,
    primary_gain_linear,
    freq_hz=150e6,
    n_ant=256,
    array_gain_mode="N",
):
    """
    Imaging-array convolution kernel.
    A source in source_cell_id is redistributed across the image by the
    normalized synthesis beam. The resulting sky map is modulated by the
    primary-beam gain in each output sky direction.
        convolved_map(c) += P(source)
                            * array_gain_factor
                            * synthesis_beam(c | source)
                            * primary_gain(c)
    """

    cell_az = np.asarray(grid_info["cell_lon"], dtype=float).reshape(-1)
    cell_el = np.asarray(grid_info["cell_lat"], dtype=float).reshape(-1)

    src_az = cell_az[source_cell_id]
    src_el = cell_el[source_cell_id]

    synth = array_factor_for_pointing(
        xyz_m=xyz_m,
        freq_hz=freq_hz,
        point_az_deg=src_az,
        point_el_deg=src_el,
        sample_az_deg=cell_az,
        sample_el_deg=cell_el,
    )

    gain_factor = get_array_gain_factor(n_ant, array_gain_mode)

    kernel = gain_factor * synth * primary_gain_linear

    return kernel


def add_mwa_sun_elevation(df, time_col="time"):
    """
    Sun up or down, for daytime/night time comparison only
    """
    mwa_location = EarthLocation(
        lat=-26.7033 * u.deg,
        lon=116.6708 * u.deg,
        height=377 * u.m,
    )

    times = Time(df[time_col].to_numpy(), format="unix")
    sun_altaz = get_sun(times).transform_to(
        AltAz(obstime=times, location=mwa_location)
    )

    out = df.copy()
    out["sun_el_deg"] = sun_altaz.alt.deg
    out["is_day"] = out["sun_el_deg"] > 0
    return out


def plot_epfd_cdf_paper(
    values_linear,
    threshold_db=-259.0,
    output_file="cdf_convolved.pdf",
    y_min=None,
):
    values_linear = np.asarray(values_linear, dtype=float)
    total_samples = len(values_linear)

    nonzero = values_linear[values_linear > 0]
    frac_zero = 1.0 - len(nonzero) / total_samples

    if len(nonzero) == 0:
        raise ValueError("No nonzero values to plot.")

    values_db = 10.0 * np.log10(nonzero)
    sorted_vals = np.sort(values_db)

    cdf_percent = (
        np.arange(1, len(sorted_vals) + 1) / total_samples
        + frac_zero
    ) * 100.0

    threshold_linear = 10.0 ** (threshold_db / 10.0)
    frac_exceed = np.mean(values_linear > threshold_linear)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_vals, cdf_percent, linewidth=2, label="Empirical CDF")

    plt.axvline(
        threshold_db,
        linestyle="--",
        linewidth=2,
        label=f"RA.769 threshold = {threshold_db:.0f} dB(W/m$^2$/Hz)",
    )

    plt.text(0.02, 0.15, f"{frac_zero*100:.1f}% of samples = 0",
             transform=plt.gca().transAxes, fontsize=10)

    plt.text(0.02, 0.08, f"Exceedance: {frac_exceed*100:.3f}%",
             transform=plt.gca().transAxes, fontsize=10)

    plt.xlabel("EPFD-like value [dB(W/m$^2$/Hz)]")
    plt.ylabel("Cumulative probability [%]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.xlim([sorted_vals.min() - 10, sorted_vals.max() + 10])

    if y_min is None:
        y_min = max(0, np.floor(frac_zero * 100) - 1)
    plt.ylim([y_min, 100.2])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

    print(f"Zero fraction: {frac_zero*100:.6f}%")
    print(f"Exceedance fraction: {frac_exceed*100:.6f}%")


def nearest_tessellation_cell(grid_info, az_deg, el_deg):
    cell_az = np.asarray(grid_info["cell_lon"], dtype=float).reshape(-1)
    cell_el = np.asarray(grid_info["cell_lat"], dtype=float).reshape(-1)

    s_req = azel_to_unit_vector(az_deg, el_deg)
    s_cells = azel_to_unit_vector(cell_az, cell_el)

    dot = np.sum(s_cells * s_req, axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    sep_deg = np.degrees(np.arccos(dot))
    idx = int(np.argmin(sep_deg))

    return idx, cell_az[idx], cell_el[idx], sep_deg[idx]


def plot_primary_synthesis_combined_polar(
    xyz_m,
    grid_info,
    primary_gain_linear,
    point_az_deg,
    point_el_deg,
    freq_hz=150e6,
    n_ant=256,
    array_gain_mode="N",
    floor_db=-80,
    marker_size=14,
):
    nearest_cell, actual_az_deg, actual_el_deg, sep_deg = nearest_tessellation_cell(
        grid_info, point_az_deg, point_el_deg
    )

    cell_az = np.asarray(grid_info["cell_lon"], dtype=float).reshape(-1)
    cell_el = np.asarray(grid_info["cell_lat"], dtype=float).reshape(-1)

    synth = array_factor_for_pointing(
        xyz_m=xyz_m,
        freq_hz=freq_hz,
        point_az_deg=actual_az_deg,
        point_el_deg=actual_el_deg,
        sample_az_deg=cell_az,
        sample_el_deg=cell_el,
    )

    gain_factor = get_array_gain_factor(n_ant, array_gain_mode)

    # Imaging-map convention:
    # combined = synthesis response × primary gain at output direction.
    combined = gain_factor * synth * primary_gain_linear

    panels = [
        ("Primary gain", primary_gain_linear),
        ("Normalized synthesis beam", synth),
        (f"{array_gain_mode}: array gain × synthesis × output primary", combined),
    ]

    theta = np.radians(cell_az)
    r = np.radians(90.0 - cell_el)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw={"projection": "polar"}
    )

    for ax, panel in zip(axes, panels):
        title = panel[0]
        values_linear = np.asarray(panel[1], dtype=float)

        vmax = np.nanmax(values_linear)
        vmax_db = 10 * np.log10(vmax)

        v_point = values_linear[nearest_cell]
        v_point_db = 10 * np.log10(v_point) if v_point > 0 else -np.inf

        values_db = 10 * np.log10(
            np.maximum(values_linear, 10 ** (floor_db / 10.0))
        )

        sc = ax.scatter(
            theta,
            r,
            c=values_db,
            s=marker_size,
            vmin=floor_db,
            vmax=vmax_db,
        )

        ax.scatter(
            np.radians(actual_az_deg),
            np.radians(90.0 - actual_el_deg),
            marker="x",
            s=120,
            linewidths=2,
            color="white",
        )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, np.pi / 2)
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

        ax.set_title(
            f"{title}\n"
            f"max = {vmax:.3e} ({vmax_db:.1f} dB)\n"
            f"@point = {v_point:.3e} ({v_point_db:.1f} dB)"
        )

        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("Power / gain [dB]")

    fig.suptitle(
        f"Beam components @ {freq_hz/1e6:.1f} MHz\n"
        f"Requested Az={point_az_deg:.1f}°, El={point_el_deg:.1f}°\n"
        f"Used cell Az={actual_az_deg:.1f}°, El={actual_el_deg:.1f}° "
        f"(Δ={sep_deg:.2f}°)",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(r"C:\Users\gregh\Desktop\EPFD\beams.pdf", dpi=300)
    plt.show()


# data set structure
cols = [
    "time", "img_idx", "freq_hz", "norad", "fchan",
    "pol", "flux_jy", "range_km",
    "ra_fit", "dec_fit", "ra_tle", "dec_tle",
    "az", "el",
    "x_fit", "y_fit", "x_tle", "y_tle",
    "pix_dist", "x_err", "y_err"
]

df = pd.read_csv(CATALOG_CSV, sep=r"\s+", names=cols, header=None)

print("Raw shape:", df.shape)

df = df[df["fchan"] == STACKED_CHANNEL].copy()
df = df[(df["freq_hz"] >= FREQ_MIN_HZ) & (df["freq_hz"] <= FREQ_MAX_HZ)].copy()

if TIME_MIN is not None:
    df = df[df["time"] >= TIME_MIN].copy()
if TIME_MAX is not None:
    df = df[df["time"] <= TIME_MAX].copy()

df = add_mwa_sun_elevation(df, time_col="time")

if DAY_NIGHT_MODE == "day":
    df = df[df["is_day"]].copy()
elif DAY_NIGHT_MODE == "night":
    df = df[~df["is_day"]].copy()
elif DAY_NIGHT_MODE != "any":
    raise ValueError("DAY_NIGHT_MODE must be 'any', 'day', or 'night'")

if POL_MODE == "XX":
    df = df[df["pol"] == 0].copy()
    pol_label = "XX"
elif POL_MODE == "YY":
    df = df[df["pol"] == 1].copy()
    pol_label = "YY"
else:
    raise ValueError("This convolution script currently expects POL_MODE = 'XX' or 'YY'.")

df = df[np.isfinite(df["flux_jy"])]
df = df[df["flux_jy"] > 0]
df = df[np.isfinite(df["az"]) & np.isfinite(df["el"])]
df = df[df["el"] >= MIN_ELEVATION_DEG].copy()

df["time_bin"] = (
    np.floor(df["time"] / SNAPSHOT_SECONDS).astype(int) * SNAPSHOT_SECONDS
).astype(int)

# No primary beam correction here
df["pfd_w_m2_hz"] = df["flux_jy"] * 1e-26

print("Shape after filtering:", df.shape)
print("Polarization:", pol_label)
print("Unique frequencies (MHz):", np.sort(df["freq_hz"].unique() / 1e6))



_, _, grid_info = sky_cells_m1583(
    niters=1,
    step_size=CELL_STEP_SIZE,
    lat_range=(0 * u.deg, 90 * u.deg),
    test=0,
)

n_cells = len(grid_info)
all_cells = np.arange(n_cells)

print("Number of sky cells:", n_cells)

df = assign_cells_from_grid(df, grid_info)
df = df[df["cell_id"] >= 0].copy()

print("Shape after cell assignment:", df.shape)


xyz_m = load_array_coordinates(COORDS_CSV)
print(f"Loaded {len(xyz_m)} array elements")

primary_df = compute_primary_gain_on_tessellation(
    grid_info=grid_info,
    freq_hz=FREQ_HZ,
    pol=POL_MODE,
    method=PRIMARY_GAIN_METHOD,
    weighted_grid_proxy=GRID_PROXY_WEIGHTED,
    delays=BEAM_DELAYS,
    amps=BEAM_AMPS,
    dipole_index=DIPOLE_INDEX,
)

primary_gain_linear = primary_df["primary_gain_linear"].to_numpy()

print(primary_df["primary_gain_dbi"].describe())


inst = (
    df.groupby(["time_bin", "cell_id"])["pfd_w_m2_hz"]
      .sum()
      .reset_index()
)

print("Number of instantaneous nonzero (time,cell) samples:", len(inst))


if OVERLAP_PERCENT < 0 or OVERLAP_PERCENT >= 100:
    raise ValueError("OVERLAP_PERCENT must be >= 0 and < 100.")

step_samples = int(round(WINDOW_SAMPLES * (1.0 - OVERLAP_PERCENT / 100.0)))
step_samples = max(step_samples, 1)

window_duration_sec = WINDOW_SAMPLES * SNAPSHOT_SECONDS
step_seconds = step_samples * SNAPSHOT_SECONDS

print(f"Window length: {WINDOW_SAMPLES} samples = {window_duration_sec} s")
print(f"Overlap: {OVERLAP_PERCENT:.1f}%")
print(f"Window step: {step_samples} samples = {step_seconds} s")

t_min = int(np.floor(df["time_bin"].min() / SNAPSHOT_SECONDS) * SNAPSHOT_SECONDS)
t_max = int(np.ceil(df["time_bin"].max() / SNAPSHOT_SECONDS) * SNAPSHOT_SECONDS)

window_starts = np.arange(
    t_min,
    t_max - window_duration_sec + SNAPSHOT_SECONDS,
    step_seconds,
    dtype=int,
)

print("Number of candidate windows:", len(window_starts))


kernel_cache = {}
convolved_windows = []

for iw, t0 in enumerate(window_starts):
    t1 = t0 + window_duration_sec

    inst_w = inst[(inst["time_bin"] >= t0) & (inst["time_bin"] < t1)]

    if inst_w.empty:
        continue

    source_avg = (
        inst_w.groupby("cell_id")["pfd_w_m2_hz"]
              .sum()
              .reindex(all_cells)
              .fillna(0.0)
              .to_numpy()
              / WINDOW_SAMPLES
    )

    nonzero_source_cells = np.where(source_avg > 0)[0]
    convolved_map = np.zeros(n_cells, dtype=float)

    for src_cell in nonzero_source_cells:
        if src_cell not in kernel_cache:
            kernel_cache[src_cell] = compute_convolution_kernel_for_source_cell(
                source_cell_id=src_cell,
                xyz_m=xyz_m,
                grid_info=grid_info,
                primary_gain_linear=primary_gain_linear,
                freq_hz=FREQ_HZ,
                n_ant=N_ANT,
                array_gain_mode=ARRAY_GAIN_MODE,
            )

        convolved_map += source_avg[src_cell] * kernel_cache[src_cell]

    out = pd.DataFrame({
        "window_index": iw,
        "window_start": t0,
        "window_stop": t1,
        "cell_id": all_cells,
        "pfd_w_m2_hz_conv": convolved_map,
    })

    convolved_windows.append(out)

    if (len(convolved_windows) % 10) == 0:
        print(f"Processed {len(convolved_windows)} active windows...")

if len(convolved_windows) == 0:
    raise ValueError("No active windows found.")

epfd_conv = pd.concat(convolved_windows, ignore_index=True)

epfd_conv["epfd_db_w_m2_hz_conv"] = np.where(
    epfd_conv["pfd_w_m2_hz_conv"] > 0,
    10.0 * np.log10(epfd_conv["pfd_w_m2_hz_conv"]),
    np.nan,
)

linear_values = epfd_conv["pfd_w_m2_hz_conv"].to_numpy()

print("Number of active windows processed:", epfd_conv["window_index"].nunique())
print("Total window×cell samples:", len(linear_values))
print("Nonzero window×cell samples:", np.sum(linear_values > 0))
print("Fraction nonzero:", 100 * np.mean(linear_values > 0), "%")

threshold_linear = 10.0 ** (RA769_THRESHOLD_DB_W_M2_HZ / 10.0)
exceedance = np.mean(linear_values > threshold_linear)

print(f"RA.769 threshold: {RA769_THRESHOLD_DB_W_M2_HZ:.1f} dB(W/m^2/Hz)")
print(f"Fraction above threshold: {100 * exceedance:.6f}%")


np.savez(fr"C:\Users\gregh\Desktop\EPFD\cdf_{POL_MODE}_{DAY_NIGHT_MODE}.npz",
         linear_values=linear_values,
         RA769_THRESHOLD_DB_W_M2_HZ=RA769_THRESHOLD_DB_W_M2_HZ,
)

plot_epfd_cdf_paper(
    linear_values,
    threshold_db=RA769_THRESHOLD_DB_W_M2_HZ,
    output_file=fr"C:\Users\gregh\Desktop\EPFD\cdf_convolved_{POL_MODE}_{ARRAY_GAIN_MODE}.pdf",
)

plot_primary_synthesis_combined_polar(
    xyz_m=xyz_m,
    grid_info=grid_info,
    primary_gain_linear=primary_gain_linear,
    point_az_deg=180.0,
    point_el_deg=60.0,
    freq_hz=FREQ_HZ,
    n_ant=N_ANT,
    array_gain_mode=ARRAY_GAIN_MODE,
    floor_db=-40,
)
