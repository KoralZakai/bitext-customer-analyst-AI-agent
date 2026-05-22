"""
One-shot: download Bitext and run preprocessing.

Name: setup_data (script)
Input: None
Output: None (delegates to src.data.preprocess.main)
Purpose: Convenience alias for python scripts/setup_data.py.
"""

from src.data.preprocess import main

if __name__ == "__main__":
    main()
