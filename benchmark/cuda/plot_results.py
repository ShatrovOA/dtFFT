import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# --- Style Settings ---
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 14

# --- Hardcoded Benchmark Data ---

# Define the order of libraries for each scaling type
strong_headers = (
    "dtFFT with Z-slab enabled",
    "dtFFT",
    "cuDECOMP",
    "HeFFTe with CUFFT",
    "AccFFT",
)
weak_headers = ("dtFFT with Z-slab enabled", "dtFFT", "cuDECOMP")

# Data format: {num_gpus: (lib1_time, lib2_time, ...)}
strong_scaling_data = {
    1: (1233, 2406, 5089, 5194, None),
    2: (10341, 10963, 11818, 22308, None),
    4: (11157, 11433, 11667, 33943, 136429),
    8: (13152, 13327, 13423, 62498, 81778),
    12: (10988, 11087, 11221, 56414, 66110),
    16: (9043, 9130, 9182, 47118, 46586),
    20: (None, 7776, 8204, 39805, 37936),
    24: (None, 6793, 7067, 34249, 32609),
    28: (None, 6034, 6253, 30260, 28131),
    32: (None, 5336, 5533, 26578, 24664),
    36: (None, 4990, 5025, 24519, 24251),
    40: (None, 4515, 4599, 21898, 21506),
}

weak_scaling_data = {
    1: (1233, 2406, 5089),
    2: (20833, 22035, 23930),
    4: (44643, 45920, 47634),
    8: (104696, 105888, 107538),
    12: (131328, 132175, 135118),
    16: (143514, 144869, 146405),
    20: (153856, 154343, 163137),
    24: (159255, 160935, 167862),
    28: (165720, 166899, 173269),
    32: (167104, 168581, 175875),
    36: (None, 173065, 179312),
    40: (173651, 175504, 182334),
}

# --- Data Processing ---


def create_dataframe(data_dict, headers):
    """Converts the dictionary data into a long-form DataFrame for Seaborn."""
    records = []
    for gpus, timings in data_dict.items():
        for i, time_ms in enumerate(timings):
            records.append({"GPUs": gpus, "Library": headers[i], "Time_ms": time_ms})
    return pd.DataFrame(records)


# Create a single DataFrame from all data
df_strong = create_dataframe(
    strong_scaling_data, strong_headers
)
df_weak = create_dataframe(weak_scaling_data, weak_headers)
dfs = [df_strong, df_weak]

for name, df, log in zip(["Strong scaling", "Weak scaling"], dfs, [True, False]):
    plt.figure()

    ax = sns.lineplot(
        data=df,
        x="GPUs",
        y="Time_ms",
        hue="Library",
        marker="o",
        linewidth=3,
        markersize=10,
    )

    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Number of GPUs", fontsize=16, fontweight="bold")
    ax.set_ylabel("Time (ms)", fontsize=16, fontweight="bold")
    ax.set_title(
        f"{name} Performance", fontsize=18, fontweight="bold", pad=20
    )

    ax.legend(
        title="Library",
        title_fontsize=14,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    filename = f"{name.lower().replace(' ', '_')}_performance.png"
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Saved to ", OUT_DIR)