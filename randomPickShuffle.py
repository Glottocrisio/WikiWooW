import csv
import random

input_filename = 'C:\\Users\\Palma\\Desktop\\PHD\\ISAAKbackup\\ISAAKx\\temp_datasets\clkstrdataset6.tsv'  # Replace with your input file name
entries = []

with open(input_filename, 'r', encoding='utf-8') as file:
    for line in file:
        # Split the line by tab and take only the first two parts
        parts = line.strip().split(';')[:2]
        if len(parts) == 2:
            entries.append('; '.join(parts))

num_entries = min(1000, len(entries))

# Randomly select entries
selected_entries = random.sample(entries, num_entries)

with open('selected_entries.txt', 'w',  encoding= 'utf-8') as f:
    for entry in selected_entries:
        f.write(entry + '\n')

with open('selected_entries.csv', 'w', encoding= 'utf-8', newline='') as f:
    writer = csv.writer(f)
    for entry in selected_entries:
        writer.writerow(entry.split(';'))

print(f"{num_entries} entries have been randomly selected and saved to 'selected_entries.txt' and 'selected_entries.csv'.")



# Input files
txt_input_filename = 'selected_entries.txt'
csv_input_filename = 'selected_entries.csv'

# Output files
txt_output_filename = 'shuffled_entries.txt'
csv_output_filename = 'shuffled_entries.csv'

# Read and shuffle TXT file
with open(txt_input_filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

lines = [line.strip() for line in lines]
random.shuffle(lines)

with open(txt_output_filename, 'w', encoding='utf-8') as file:
    for line in lines:
        file.write(line + '\n')

#Read and shuffle CSV file
with open(csv_input_filename, 'r', encoding='utf-8', newline='') as file:
    reader = csv.reader(file)
    rows = list(reader)

random.shuffle(rows)

with open(csv_output_filename, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print(f"Lines have been shuffled and written to '{txt_output_filename}'") # and '{csv_output_filename}'.")