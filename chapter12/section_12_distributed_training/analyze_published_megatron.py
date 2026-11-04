"""Plot actual SC21 Megatron-LM Table 1 published benchmarks. Not a new experiment."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parent
    d = pd.read_csv(root / "megatron_sc21_table1.csv").sort_values("gpus")
    required = ["model_parameters_billion", "gpus", "tflops_per_gpu",
                "percent_of_theoretical_peak", "aggregate_pflops"]
    assert len(d) == 10 and all(c in d for c in required)
    assert d.gpus.is_unique
    assert np.allclose(d.gpus*d.tflops_per_gpu/1000, d.aggregate_pflops,
                       rtol=.02, atol=.1)
    out = root / "output"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    ax[0].plot(d.gpus, d.aggregate_pflops, "o-", color="#214D87", linewidth=2)
    ax[0].set(xlabel="NVIDIA A100 GPUs", ylabel="Aggregate throughput (PFLOP/s)")
    ax[1].plot(d.model_parameters_billion, d.percent_of_theoretical_peak,
               "o-", color="#157B78", linewidth=2)
    ax[1].set(xlabel="Model size (billions of parameters)",
              ylabel="Achieved theoretical peak (%)")
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "megatron_sc21_published_measurements.png", bbox_inches="tight")
    fig.savefig(out / "megatron_sc21_published_measurements.pdf", bbox_inches="tight")
    d.to_csv(out / "verified_published_measurements.csv", index=False)
    print("Source: Narayanan et al., SC21 (2021), Table 1; published measurements")
    print(d.to_string(index=False))

if __name__ == "__main__":
    main()
