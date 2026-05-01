import os
import numpy as np
import matplotlib.pyplot as plt


def plot_epfd_cdf_day_night_for_pol(
    pol_mode,
    base_dir=r"C:\Users\gregh\Desktop\EPFD",
    threshold_db=-259.0,
    output_file=None,
    modes=("any", "day", "night"),
    labels=None,
    y_min=None,
):
    """
    Plot CDFs for one polarization, comparing any/day/night.
    Expects files named:
        cdf_XX_any.npz
        cdf_XX_day.npz
        cdf_XX_night.npz
    or same for YY.
    """

    if labels is None:
        labels = {
            "any": "All data",
            "day": "Day",
            "night": "Night",
        }

    curves = []

    for mode in modes:
        path = os.path.join(base_dir, f"cdf_{pol_mode}_{mode}.npz")

        if not os.path.exists(path):
            print(f"Missing file, skipping: {path}")
            continue

        data = np.load(path)
        values_linear = np.asarray(data["linear_values"], dtype=float)

        if "RA769_THRESHOLD_DB_W_M2_HZ" in data:
            threshold_db = float(data["RA769_THRESHOLD_DB_W_M2_HZ"])

        total_samples = len(values_linear)
        nonzero = values_linear[values_linear > 0]

        if len(nonzero) == 0:
            print(f"No nonzero samples for {pol_mode} / {mode}, skipping.")
            continue

        frac_zero = 1.0 - len(nonzero) / total_samples

        values_db = 10.0 * np.log10(nonzero)
        sorted_vals = np.sort(values_db)

        cdf_percent = (
            np.arange(1, len(sorted_vals) + 1) / total_samples
            + frac_zero
        ) * 100.0

        threshold_linear = 10.0 ** (threshold_db / 10.0)
        frac_exceed = np.mean(values_linear > threshold_linear)

        curves.append({
            "mode": mode,
            "label": labels.get(mode, mode),
            "sorted_vals": sorted_vals,
            "cdf_percent": cdf_percent,
            "frac_zero": frac_zero,
            "frac_exceed": frac_exceed,
            "n_total": total_samples,
            "n_nonzero": len(nonzero),
        })

    if len(curves) == 0:
        raise ValueError(f"No valid CDF files found for polarization {pol_mode}")

    plt.figure(figsize=(8, 6))

    for c in curves:
        plt.plot(
            c["sorted_vals"],
            c["cdf_percent"],
            linewidth=2,
            label=(
                f"{c['label']} "
                f"(exceed. {100*c['frac_exceed']:.2f}%)"
            )
        )

    plt.axvline(
        threshold_db,
        linestyle="--",
        linewidth=2,
        label=f"RA.769 threshold = {threshold_db:.0f} dB(W/m$^2$/Hz)",
    )

    plt.xlabel("EPFD-like value [dB(W/m$^2$/Hz)]")
    plt.ylabel("Cumulative probability [%]")
    plt.title(f"EPFD-like CDF, {pol_mode} polarization")
    plt.grid(True, alpha=0.3)
    plt.legend()

    all_x = np.concatenate([c["sorted_vals"] for c in curves])
    plt.xlim([np.nanmin(all_x) - 10, np.nanmax(all_x) + 10])

    if y_min is None:
        min_zero = min(c["frac_zero"] for c in curves)
        y_min = max(0, np.floor(min_zero * 100) - 1)

    plt.ylim([y_min, 100.2])

    plt.tight_layout()

    if output_file is None:
        output_file = os.path.join(base_dir, f"cdf_{pol_mode}_day_night_comparison.pdf")

    plt.savefig(output_file, dpi=300)
    plt.show()

    for c in curves:
        print(
            f"{pol_mode} / {c['mode']}: "
            f"total={c['n_total']}, "
            f"nonzero={c['n_nonzero']}, "
            f"zero={100*c['frac_zero']:.6f}%, "
            f"exceedance={100*c['frac_exceed']:.6f}%"
        )

plot_epfd_cdf_day_night_for_pol("XX", y_min=0)
plot_epfd_cdf_day_night_for_pol("YY", y_min=0)