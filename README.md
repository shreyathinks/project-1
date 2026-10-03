# GraphGuard

GraphGuard is an adversarial, time-aware oversampling framework for minority-class fraud detection on the Elliptic Bitcoin dataset.
It utilizes a dual-head structural discriminator and strict temporal edge injection to generate realistic illicit transaction behavior.

## How to Run

1. **Install `uv` (if not already installed):**
   - **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   *(Alternatively, you can just use `pip install uv`)*

2. **Install Dependencies:**
```bash
uv sync
```

3. **Execute Pipeline:**
```bash
uv run scripts/01_eda_and_preprocessing.py
uv run scripts/02_train_generator.py
uv run scripts/03_train_classifier.py
```
