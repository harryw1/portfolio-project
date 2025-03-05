from pathlib import Path
import sys
from src import module

# Add the parent directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    module.test()
