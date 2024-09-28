# I finetuned the model on my old college essays.
# I will not be open sourcing that datset.

import os
from docx import Document
from PyPDF2 import PdfReader

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    
    for paragraph in doc.paragraphs:
        # Check if the paragraph is part of a footnote
        if not any(paragraph.style.name.startswith(style_name) for style_name in ["Footnote", "Endnote"]):
            full_text.append(paragraph.text)
    
    return '\n'.join(full_text)

def combine_text_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith('.docx'):
                file_path = os.path.join(directory, filename)
                text = extract_text_from_docx(file_path)
                outfile.write(text + '\n')

# Specify the directory and output file
directory = './Downloads/All Final Drafts'
output_file = './TrainingLLMFromScratchMLX/combined_essays.txt'

# Run the function
combine_text_files(directory, output_file)

def format_text(file_path, max_line_length=70, output_path=None):
    if output_path is None:
        output_path = file_path  # Overwrite the original file if no output path is provided

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    formatted_lines = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        while len(line) > max_line_length:
            split_point = line.rfind(' ', 0, max_line_length)
            if split_point == -1:  # No space found, force split
                split_point = max_line_length
            formatted_lines.append(line[:split_point])
            line = line[split_point:].strip()
        if line:
            formatted_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(formatted_lines) + '\n')

# Specify the file path
file_path = './TrainingLLMFromScratchMLX/combined_essays.txt'

# Format the text with lines no longer than 70 characters
format_text(file_path)