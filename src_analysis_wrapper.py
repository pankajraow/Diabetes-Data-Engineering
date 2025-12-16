import sys
import os

# Redirect stdout to a file to capture the summary stats table
sys.stdout = open(os.path.join("output", "summary_output.txt"), "w")

# Import original script main
sys.path.append("src")
import analysis

if __name__ == "__main__":
    analysis.main()
