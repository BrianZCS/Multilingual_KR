# Function to filter out English alphabetic characters and newline characters while keeping Chinese characters, numbers, and other symbols
# Additionally, group every three sentences into one line based on periods (full stops)
def filter_file_and_group(input_file, output_file):
    import re

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        buffer = []  # Buffer to hold sentences
        for line in infile:
            # Remove English letters and newline characters but keep Chinese characters, numbers, and other symbols
            filtered_line = ''.join(char for char in line if not (char.isascii() and char.isalpha()) and char != '\n')
            # Split the line into sentences based on periods (full stops)
            sentences = re.split(r'(?<=\u3002)', filtered_line)  # '\u3002' is the Unicode for Chinese period
            for sentence in sentences:
                if sentence.strip():  # Ignore empty sentences
                    buffer.append(sentence)
                # Write out every three sentences as one line
                if len(buffer) == 3:
                    outfile.write(''.join(buffer) + '\n')
                    buffer = []
        
        # Write remaining sentences in the buffer
        if buffer:
            outfile.write(''.join(buffer) + '\n')

# Example usage
input_file = 'wiki_zh1.txt'  # Replace with the path to your input file
output_file = 'wiki_zh.txt'  # Replace with the desired path for the output file

filter_file_and_group(input_file, output_file)

print(f"Filtered and grouped content saved to {output_file}")
#chatgpt