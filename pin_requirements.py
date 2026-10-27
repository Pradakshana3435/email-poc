#pin_requirements.py
"""Write requirements.txt with the exact versions installed in the current environment.

Run it inside the virtual environment you trained and tested with:
    python pin_requirements.py
Only the packages the project imports are pinned (not the hundreds of
dependencies `pip freeze` would list); pip resolves the rest.
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = [
    "pandas", "numpy", "scipy", "scikit-learn", "sentence-transformers",
    "joblib", "PyYAML", "streamlit", "matplotlib",
]

lines = [
    f"# Pinned from the environment that built and tested this project (Python {sys.version.split()[0]}).",
]
missing = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        missing.append(name)
        lines.append(name)

Path("requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
if missing:
    print("\nNot installed here (left unpinned):", ", ".join(missing))
