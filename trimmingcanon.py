# This file gives examples for how I trimmed down the canon
# This was largely based on heuristics. I eyeballed what looked like noise and what 
# I mainly eliminated licenses, emails, website names, and so on.
# These books are all open source so I am free to strip the licenses.
# But according to their license, I am not free to mention where they come from as the name of the host organization is copyrighted.
# This code is extremely messy.

import os

# Set the directory path and the output file path
directory_path = "./PlatoV1/PlatoV1Dataset/JustBooks/WesternCanon"
output_file = "./TrainingLLMFromScratchMLX/WesternCanon_combined.txt"

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Appending {file_path} to {output_file}")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        # Append the content of each file to the output file
                        outfile.write(infile.read())
                        # Optionally, add a newline between files
                        outfile.write("\n\n")
                except UnicodeDecodeError as e:
                    print(f"Skipping {file_path} due to encoding error: {e}")

print(f"All files have been combined into {output_file}")

# Define the path to the text file
file_path = "./TrainingLLMFromScratchMLX/WesternCanon_combined.txt"

# Define the range of lines to delete
start_line = 356216
end_line = 409848

# Read the file, delete the specified lines, and write the result back to the file
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Delete the lines in the specified range
del lines[start_line-1:end_line]

# Write the modified content back to the file
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(lines)

print(f"Lines {start_line} through {end_line} have been deleted from {file_path}.")

import re

# Define the path to the text file
file_path = "./TrainingLLMFromScratchMLX/WesternCanon_combined.txt"

# Read the contents of the file
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Define a regular expression pattern to match sequences of more than 3 newlines (with possible tabs/spaces)
pattern = r'(\s*\n\s*){4,}'

# Replace those sequences with exactly 3 newlines
modified_content = re.sub(pattern, '\n\n\n', content)

# Write the modified content back to the file
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(modified_content)

print(f"Excessive whitespace has been normalized in {file_path}.")


import os

# Define the path to the text file
file_path = "./TrainingLLMFromScratchMLX/WesternCanon_combined.txt"

# All licenses will be added to the Hugging Face repository
string_to_remove = """String to remove"""

strings_to_remove = [
        """To protect the COMPANYNAME mission of promoting the free
distribution of electronic works, by using or distributing this work
(or any other work associated in any way with the phrase “COMPANYNAME”),
you agree to comply with all the terms of the Full COMPANYNAME License
(available with this file or online at
http://companywebsite.org/license).""",

    """1.B.  “COMPANYNAME” is a registered trademark.  It may only be
used on or associated in any way with an electronic work by people who
agree to be bound by the terms of this agreement.""",

]

# Read the contents of the file
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Replace all instances of the string with an empty string
modified_content = content.replace(string_to_remove, 'COMPANYNAME')

# Write the modified content back to the file
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(modified_content)

print(f"All instances of the specified string have been removed from {file_path}.")