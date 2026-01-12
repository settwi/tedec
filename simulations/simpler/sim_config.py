from yaff import fitting

directories = {
    "cold": "cold-flare",
    "typical": "typical-flare",
    "weak nonthermal": "weak nonthermal"
}
rev_directories = {v: k for (k, v) in directories.items()}

# Energy fit ranges bounds per-simulation
energy_ranges = {
    "cold": {
        "traditional": [3, 70],
        "thermal": [3, 10],
        "nonthermal": [7, 90],
    },
    "typical": {
        "thermal": [3, 16],
        "nonthermal": [10, 100],
        "traditional": [3, 70],
    },
    "weak nonthermal": {
        "traditional": [3, 70],
        "thermal": [3, 16],
        "nonthermal": [3, 80],
    },
}

# Parameter bounds per-simulation
priors = {
    "cold": {
        "thermal": {
            "temperature": fitting.simple_bounds(1, 40),
            "emission_measure": fitting.simple_bounds(1e-4, 1e4),
        },
        "nonthermal": {
            "electron_flux": fitting.simple_bounds(0, 20),
            "spectral_index": fitting.simple_bounds(2, 20),
            "cutoff_energy": fitting.simple_bounds(1, 80),
        },
    },
    "typical": {
        "thermal": {
            "temperature": fitting.simple_bounds(10, 40),
            "emission_measure": fitting.simple_bounds(1e-4, 1e4),
        },
        "nonthermal": {
            "electron_flux": fitting.simple_bounds(0, 20),
            "spectral_index": fitting.simple_bounds(2, 20),
            "cutoff_energy": fitting.simple_bounds(1, 80),
        },
    },
    "weak nonthermal": {
        "thermal": {
            "temperature": fitting.simple_bounds(1, 40),
            "emission_measure": fitting.simple_bounds(1e-4, 1e4),
        },
        "nonthermal": {
            "electron_flux": fitting.simple_bounds(0, 20),
            "spectral_index": fitting.simple_bounds(2, 20),
            "cutoff_energy": fitting.simple_bounds(1, 80),
        },
    },
}