import os

# Set folder path and output file
folder_path = "/home/dooseop/DATASET/ETRI/av2format/test_flops_qcnet"
output_file = 'test_flops_list.txt'

# Filter for .pkl files
pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

# Write to text file
with open(output_file, 'w') as f:
    for file in pkl_files:
        f.write(file + '\n')

print(f"PKL file names written to {output_file}")