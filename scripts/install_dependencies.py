"""Install dependencies for the Colab runtime."""

import subprocess
import sys

PACKAGES = [
    "ultralytics",
    "opencv-python-headless",
    "pandas",
    "pyyaml",
    "tqdm",
    "boxmot",
    "scipy"
]


def main() -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + PACKAGES
    print("Installing:", " ".join(PACKAGES))
    subprocess.check_call(cmd)
    print("Dependency installation complete.")


if __name__ == "__main__":
    main()