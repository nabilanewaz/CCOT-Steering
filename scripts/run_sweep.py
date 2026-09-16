"""Run the complete full-data pipeline through the canonical entry point."""
import subprocess
import sys


def main():
    subprocess.run([sys.executable, "pipeline.py", "--phase", "0", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
