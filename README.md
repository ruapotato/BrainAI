# BrainAI

BrainAI is an AI-powered file system navigator and question-answering system that combines the functionality of a file explorer with the capabilities of a large language model.

## Author

David Hamner

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

## Features

- Navigate and explore file systems
- Process various file types (text, images, PDFs, Word documents)
- Ask questions about file contents and receive AI-generated responses
- Search for files based on content
- Create, edit, and delete files through a command-line interface

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- PIL (Python Imaging Library)
- PyPDF2
- python-docx

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/brain-ai.git
   cd brain-ai
   ```

2. Create a virtual environment:
   ```
   python3 -m venv pyenv
   ```

3. Activate the virtual environment:
   - On Unix or MacOS:
     ```
     source pyenv/bin/activate
     ```
   - On Windows:
     ```
     pyenv\Scripts\activate
     ```

4. Install the required packages:
   ```
   pip install torch transformers Pillow PyPDF2 python-docx
   ```

## Usage

1. Ensure your virtual environment is activated.

2. Run the main script:
   ```
   python main.py
   ```

3. Use the following commands in the interactive prompt:
   - `cd <directory>`: Change directory
   - `ls`: List contents of current directory
   - `search <query>`: Search for files containing the query
   - `create <filename> <content>`: Create a new file (Just use a file manager and a text editor)
   - `edit <filename> <new_content>`: Edit an existing file
   - `delete <filename>`: Delete a file
   - Any other input will be treated as a question for the AI


## Contributing

Contributions to BrainAI are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This project uses a large language model and may produce unexpected or biased outputs. Use with caution and always verify important information.
