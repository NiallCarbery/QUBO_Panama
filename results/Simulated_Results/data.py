import re

# Initialize empty lists for storing the extracted values
sampling_times = []  # To store the sampling time values as floats
feasible_solutions = []  # To store the number of feasible solutions as integers

# Compile regular expressions for performance
# This pattern looks for lines like "Sampling took 0.28 seconds."
sampling_pattern = re.compile(r"Sampling took ([\d.]+) seconds\.")
# This pattern looks for lines like "Number of feasible solutions: 50"
feasible_pattern = re.compile(r"Number of feasible solutions:\s*(\d+)")

# Open and read the file line-by-line
with open("RUN1.txt", "r") as file:
    for line in file:
        # Try to find a sampling time in the current line
        sampling_match = sampling_pattern.search(line)
        if sampling_match:
            # Convert the captured string to a float and append to the list
            sampling_times.append(float(sampling_match.group(1)))

        # Try to find the number of feasible solutions in the current line
        feasible_match = feasible_pattern.search(line)
        if feasible_match:
            # Convert the captured string to an integer and append to the list
            feasible_solutions.append(int(feasible_match.group(1)))

# Output the extracted lists
print("Sampling times:", sampling_times)
print("Number of feasible solutions:", feasible_solutions)
