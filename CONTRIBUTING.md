# Contributing to Dataset Detective

Thank you for your interest in contributing to Dataset Detective! We welcome contributions, bug reports, and feature suggestions.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dataset-detective.git
   cd dataset-detective
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev,baseline]"
   ```

## Running the Test Suite

Run the full test suite with `pytest`:

```bash
pytest -v
```

## Generating Demo Datasets

You can use the built-in generator to create test datasets with intentional anomalies (imbalance, duplicates, train/test leakage):

```bash
python make_sample_dataset.py --out sample_data
```

## Pull Request Guidelines

1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Write clean, readable code and maintain documentation.
3. Add unit tests in `tests/` for any new functionality or bug fixes.
4. Ensure all tests pass locally before pushing.
5. Open a Pull Request on GitHub describing your changes.

