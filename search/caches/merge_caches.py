import json
import sys
from tqdm import tqdm  # Add tqdm import

def print_help():
    print("Usage: python merge_caches.py <input_file1> <input_file2> ... <output_file>")
    print("Merges multiple JSON files into a single JSON file.")
    print("Arguments:")
    print("  <input_file1> <input_file2> ... : JSON files to be merged")
    print("  <output_file>                  : File where the merged JSON will be saved")

if len(sys.argv) < 3 or '--help' in sys.argv or '-h' in sys.argv:
    print_help()
    sys.exit(1)

curr_data = {}

# Add tqdm to the for loop to show progress
for file in tqdm(sys.argv[1:-1], desc="Merging files"):  # Exclude the last argument (output file)
    with open(file, "r") as f:
        data = json.load(f)
        curr_data.update(data)

output_file = sys.argv[-1]  # The last argument is the output file
with open(output_file, "w") as f:
    json.dump(curr_data, f, indent=2)