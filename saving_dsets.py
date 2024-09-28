import tiktoken
import os
import mlx.data as dx
import mlx.core as mx
import os
from datasets import Dataset
from datasets import load_dataset
from gpt_layer import GPT
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from functools import partial

def concatenate_txt_files(directory):
    combined_text = ""

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                # Attempt to read the file with utf-8 encoding
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        combined_text += f.read() + "\n\n"
                
                # If a UnicodeDecodeError occurs, try another encoding
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='ISO-8859-1') as f:
                            combined_text += f.read() + "\n\n"
                    except Exception as e:
                        print(f"Failed to read {file_path} due to {str(e)}")

    return combined_text

directory = "./PlatoV1/PlatoV1Dataset/JustBooks"
combined_text = concatenate_txt_files(directory)

# Specify the output file name
output_file = "TheEntireWesternCanon.txt"

# Write the combined text to the file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(combined_text)

print(f"Combined text has been saved to {output_file}")