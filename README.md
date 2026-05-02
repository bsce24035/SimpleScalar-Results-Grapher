# SimpleScalar-Results-Grapher
A Python tool that parses SimpleScalar sim-outorder result files and generates graphs for CPU architecture analysis — covering cache performance, branch prediction, and pipeline statistics.

---

## Setup

**Requirements:** Python 3 (Linux users may need to run `sudo apt install python3-full` first)

Follow the instructions below for your specific operating system to install the required libraries (`matplotlib` and `numpy`).

### Windows

1. Open your terminal (Command Prompt or PowerShell) and run:
```bash
pip install matplotlib numpy
```
2. If you have multiple Python versions installed, use:
```bash
python -m pip install matplotlib numpy
```

### Linux

Linux distributions block global `pip` installs by default to prevent breaking system tools. Choose one of the two methods below:

#### Method A: Using a Virtual Environment
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the packages
pip install matplotlib numpy
```

#### Method B: Using System Package Manager
```bash
sudo apt update
sudo apt install python3-matplotlib python3-numpy
```
## How to Run

**Interactive mode** — prompts you for everything:

```bash
python simscalar_grapher.py
```

**Pass files directly** via command line:

```bash
python simscalar_grapher.py simresults-small.txt simresults-large.txt
```

---

## What It Does

1. Parses one or more sim-outorder result files automatically
2. Asks you to label each file (e.g. Small / Medium / Large)
3. Shows a menu of 25 graph types — pick any or generate all
4. Saves graphs as `.png` files in your chosen output folder

Covers execution time, CPI/IPC, cache miss rates, branch predictor stats, and pipeline occupancy.

---
