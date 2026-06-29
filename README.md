# LILA Mock Data

Generate mock time-domain datasets for the Laser Interferometer Lunar Antenna
(LILA). The current model projects PyCBC barycentric gravitational-wave
polarizations onto a lunar-surface, long-wavelength Michelson response using
`lunarsky` coordinate transforms.

## Scope

This repository contains production-oriented code for generating geometric LILA
mock data. It intentionally does not include the scratch notebook, generated
HDF5 outputs, logs, or Python cache files.

Current model limitations:

- no LILA noise power spectral density,
- no lunar elastic or normal-mode transfer function,
- no relativistic clock corrections,
- long-wavelength Michelson detector response only.

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The project depends on `pycbc`, `lunarsky`, `astropy`, `scipy`, `numpy`, and
`h5py`. Some scientific Python dependencies may require system libraries on a
fresh machine.

## Usage

After installation, run:

```bash
lila-mock-data --output lila_signal.hdf5
```

You can also run the wrapper directly from the repository:

```bash
python run_lila_mock_data.py --output lila_signal.hdf5
```

Useful options:

```bash
lila-mock-data \
  --mass1 1000 \
  --mass2 1000 \
  --distance 1000 \
  --sample-rate 4 \
  --duration 86400 \
  --f-lower 0.05 \
  --ra 1.3 \
  --dec 0.4 \
  --psi 0.2 \
  --debug
```

Run `lila-mock-data --help` for the full option list.

## Output

The output HDF5 file contains:

- `/barycenter/time`: barycentric time grid in seconds,
- `/barycenter/hp`: plus polarization on the barycentric grid,
- `/barycenter/hc`: cross polarization on the barycentric grid,
- `/detector/time`: detector-frame time grid in seconds,
- `/detector/hp_shifted`: delay-shifted plus polarization,
- `/detector/hc_shifted`: delay-shifted cross polarization,
- `/detector/delay`: detector-to-barycenter light-travel delay in seconds,
- `/detector/Fp`: plus antenna pattern,
- `/detector/Fc`: cross antenna pattern,
- `/detector/h_det`: projected detector strain.

Run parameters and model metadata are stored as HDF5 file attributes.

## Development

Install development dependencies with:

```bash
python -m pip install -e ".[dev]"
```

Lint the code with:

```bash
ruff check .
```

Generated products such as `*.hdf5`, `*.log`, notebook checkpoints, and cache
directories are ignored by git.

