import os
import numpy as np
import matplotlib.pyplot as plt


def load_cdf_curve(path, threshold_db=-259.0):
    data = np.load(path)
    values_linear = np.asarray(data["linear_values"], dtype=float)

    if "RA769_THRESHOLD_DB_W_M2_HZ" in data:
        threshold_db = float(data["RA769_THRESHOLD_DB_W_M2_HZ"])

    total_samples = len(values_linear)
    nonzero = values_linear[values_linear > 0]

    if len(nonzero) == 0:
        raise ValueError(f"No nonzero samples in {path}")

    frac_zero = 1.0 - len(nonzero) / total_samples

    values_db = 10.0 * np.log10(nonzero)
    sorted_vals = np.sort(values_db)

    cdf_percent = (
        np.arange(1, len(sorted_vals) + 1) / total_samples
        + frac_zero
    ) * 100.0

    threshold_linear = 10.0 ** (threshold_db / 10.0)
    frac_exceed = np.mean(values_linear > threshold_linear)

    return {
        "sorted_vals": sorted_vals,
        "cdf_percent": cdf_percent,
        "frac_zero": frac_zero,
        "frac_exceed": frac_exceed,
        "n_total": total_samples,
        "n_nonzero": len(nonzero),
        "threshold_db": threshold_db,
    }


def plot_epfd_cdf_day_night_elev_compare(
    pol_mode,
    base_dir=r"C:\Users\gregh\Desktop\EPFD",
    threshold_db=-259.0,
    output_file=None,
    modes=("any", "day", "night"),
    y_min=0,
):
    """
    Compare:
      - normal EPFD CDFs (solid)
      - elevation-cut EPFD CDFs (dashed)

    Expected files:
        cdf_XX_any.npz
        cdf_XX_any_elev_40.0.npz
        etc.
    """

    mode_labels = {
        "any": "All data",
        "day": "Day",
        "night": "Night",
    }

    mode_colors = {
        "any": "C0",
        "day": "C1",
        "night": "C2",
    }

    curves = []

    for mode in modes:

        # --------------------------------------------------
        # Normal
        # --------------------------------------------------

        normal_path = os.path.join(
            base_dir,
            f"cdf_{pol_mode}_{mode}.npz"
        )

        if os.path.exists(normal_path):

            c = load_cdf_curve(normal_path, threshold_db)

            c["mode"] = mode
            c["label"] = mode_labels[mode]
            c["linestyle"] = "-"
            c["color"] = mode_colors[mode]
            c["tag"] = "normal"

            curves.append(c)

            threshold_db = c["threshold_db"]

        else:
            print(f"Missing file: {normal_path}")

        # --------------------------------------------------
        # Elevation-cut
        # --------------------------------------------------

        elev_path = os.path.join(
            base_dir,
            f"cdf_{pol_mode}_{mode}_elev_40.0.npz"
        )

        if os.path.exists(elev_path):

            c = load_cdf_curve(elev_path, threshold_db)

            c["mode"] = mode
            c["label"] = (
                mode_labels[mode]
                + r" (El $\geq 40^\circ$)"
            )
            c["linestyle"] = "--"
            c["color"] = mode_colors[mode]
            c["tag"] = "elev40"

            curves.append(c)

            threshold_db = c["threshold_db"]

        else:
            print(f"Missing file: {elev_path}")

    if len(curves) == 0:
        raise ValueError(f"No valid curves found for {pol_mode}")

    # ======================================================
    # Plot
    # ======================================================

    plt.figure(figsize=(8, 6))

    for c in curves:

        plt.plot(
            c["sorted_vals"],
            c["cdf_percent"],
            linestyle=c["linestyle"],
            color=c["color"],
            linewidth=2,
            label=(
                f"{c['label']} "
                f"(exceed. {100*c['frac_exceed']:.1f}%)"
            ),
        )

    plt.axvline(
        threshold_db,
        linestyle=":",
        linewidth=2,
        color="k",
        label=f"RA.769 = {threshold_db:.0f} dB(W/m$^2$/Hz)",
    )

    plt.xlabel("EPFD-like value [dB(W/m$^2$/Hz)]")
    plt.ylabel("Cumulative probability [%]")

    plt.title(
        f"EPFD-like CDF comparison, {pol_mode} polarization"
    )

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    all_x = np.concatenate([c["sorted_vals"] for c in curves])

    plt.xlim([
        np.nanmin(all_x) - 10,
        np.nanmax(all_x) + 10,
    ])

    plt.ylim([y_min, 100.2])

    plt.tight_layout()

    if output_file is None:

        output_file = os.path.join(
            base_dir,
            f"cdf_{pol_mode}_day_night_elev_compare.pdf"
        )

    plt.savefig(output_file, dpi=300)
    plt.show()

    # ======================================================
    # Summary
    # ======================================================

    print("\nSummary\n" + "-" * 60)

    for c in curves:

        print(
            f"{pol_mode:>2s} | "
            f"{c['mode']:>5s} | "
            f"{c['tag']:>7s} | "
            f"zero={100*c['frac_zero']:.3f}% | "
            f"exceed={100*c['frac_exceed']:.3f}%"
        )

    print(f"\nSaved: {output_file}")


# ==========================================================
# Run
# ==========================================================

plot_epfd_cdf_day_night_elev_compare("XX", y_min=0)
plot_epfd_cdf_day_night_elev_compare("YY", y_min=0)
