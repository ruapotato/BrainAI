import os
import sys
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import re
import cmd
import logging
from io import BytesIO
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class BrainAI:
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        try:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            sys.exit(1)
        
        self.brain_root = os.path.abspath("./brain")
        self.current_path = self.brain_root
        self.context = []
        self.update_context()

    def change_directory(self, path):
        if path == "..":
            new_path = os.path.dirname(self.current_path)
        else:
            new_path = os.path.normpath(os.path.join(self.current_path, path))
        
        if os.path.commonpath([new_path, self.brain_root]) != self.brain_root:
            logging.warning(f"Cannot access directory outside of {self.brain_root}")
            return
        
        if os.path.exists(new_path) and os.path.isdir(new_path):
            self.current_path = new_path
            logging.info(f"Changed directory to: {self.get_relative_path(self.current_path)}")
            self.update_context()
        else:
            logging.warning(f"Directory not found: {self.get_relative_path(new_path)}")

    def get_relative_path(self, path):
        return os.path.relpath(path, start=self.brain_root)

    def list_directory(self):
        relative_path = self.get_relative_path(self.current_path)
        logging.info(f"Contents of {relative_path}:")
        for item in os.listdir(self.current_path):
            if os.path.isdir(os.path.join(self.current_path, item)):
                logging.info(f"📁 {item}")
            else:
                logging.info(f"📄 {item}")

    def update_context(self):
        self.context = []
        for item in os.listdir(self.current_path):
            item_path = os.path.join(self.current_path, item)
            if os.path.isdir(item_path):
                self.context.append({"type": "directory", "name": item, "path": self.get_relative_path(item_path)})
            else:
                processed_item = self.process_file(item_path)
                if processed_item:
                    self.context.append(processed_item)
        logging.debug(f"Updated context: {self.context}")

    def process_file(self, file_path):
        _, ext = os.path.splitext(file_path.lower())
        try:
            if ext in ('.txt', '.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return {"type": "text", "content": f.read(), "path": self.get_relative_path(file_path)}
            elif ext in ('.png', '.jpg', '.jpeg'):
                return {"type": "image", "content": file_path, "path": self.get_relative_path(file_path)}
            elif ext == '.pdf':
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return {"type": "text", "content": text, "path": self.get_relative_path(file_path)}
            elif ext in ('.doc', '.docx'):
                return {"type": "document", "path": self.get_relative_path(file_path)}
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
        return None

    def suggest_navigation(self, query):
        query_words = set(re.findall(r'\w+', query.lower()))
        suggestions = []

        for item in self.context:
            if item['type'] == 'directory':
                dir_words = set(re.findall(r'\w+', item['name'].lower()))
                if query_words & dir_words:
                    suggestions.append(f"cd {item['name']}")
            elif item['type'] in ['text', 'image', 'document']:
                file_words = set(re.findall(r'\w+', item['path'].lower()))
                if query_words & file_words:
                    suggestions.append(f"Examine the file: {item['path']}")

        return suggestions

    def ask_question(self, question):
        logging.info(f"Received question: {question}")
        context_text = self.summarize_context()
        context_images = self.get_context_images()

        logging.debug(f"Context text: {context_text}")
        logging.debug(f"Number of context images: {len(context_images)}")

        navigation_suggestions = self.suggest_navigation(question)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"Context: {context_text}"},
                {"type": "text", "text": f"Question: {question}"}
            ]}
        ]

        if context_images:
            for image in context_images:
                messages[0]['content'].insert(0, {"type": "image"})

        try:
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            if context_images:
                inputs = self.processor(images=context_images, text=input_text, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.processor(text=input_text, return_tensors="pt").to(self.model.device)

            output = self.model.generate(**inputs, max_new_tokens=200)
            response = self.processor.decode(output[0], skip_special_tokens=True)

            if navigation_suggestions:
                response += "\n\nBased on your question, you might want to explore the following:"
                for suggestion in navigation_suggestions:
                    response += f"\n- {suggestion}"
                response += "\nUse 'ls' after changing directories to see the contents."

            if not navigation_suggestions and ("I don't have enough information" in response or "I need more context" in response):
                response += "\n\nTo find more information, you can:"
                response += "\n- Use 'ls' to list the contents of the current directory."
                response += "\n- Use 'cd <directory_name>' to navigate to a subdirectory."
                response += "\n- Use 'cd ..' to go up one level in the directory structure."

            return response
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            return f"An error occurred while processing your question: {e}"

    def summarize_context(self):
        summary = f"Current directory: {self.get_relative_path(self.current_path)}\n"
        summary += "Contents:\n"
        for item in self.context:
            if item['type'] == 'directory':
                summary += f"- Directory: {item['name']}\n"
            elif item['type'] == 'text':
                summary += f"- File: {item['path']}\nContent: {item['content'][:500]}...\n"
            elif item['type'] == 'image':
                summary += f"- Image: {item['path']}\n"
            elif item['type'] == 'document':
                summary += f"- Document: {item['path']}\n"
        return summary

    def get_context_images(self):
        images = []
        for item in self.context:
            if item['type'] == 'image':
                try:
                    with open(item['content'], 'rb') as img_file:
                        img_bytes = BytesIO(img_file.read())
                    img = Image.open(img_bytes).convert('RGB')
                    images.append(img)
                    logging.debug(f"Successfully loaded image: {item['path']}")
                except Exception as e:
                    logging.error(f"Error opening image {item['content']}: {e}")
        return images

    def search(self, query):
        results = []
        for item in self.context:
            if item['type'] == 'text' and 'content' in item and re.search(query, item['content'], re.IGNORECASE):
                results.append(item['path'])
            elif item['type'] in ['document', 'image'] and re.search(query, item['path'], re.IGNORECASE):
                results.append(item['path'])
        return results

    def create_file(self, filename, content):
        file_path = os.path.join(self.current_path, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"File created: {self.get_relative_path(file_path)}")
            self.update_context()
        except Exception as e:
            logging.error(f"Error creating file {filename}: {e}")

    def edit_file(self, filename, new_content):
        file_path = os.path.join(self.current_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logging.info(f"File updated: {self.get_relative_path(file_path)}")
                self.update_context()
            except Exception as e:
                logging.error(f"Error editing file {filename}: {e}")
        else:
            logging.warning(f"File not found: {self.get_relative_path(file_path)}")

    def delete_file(self, filename):
        file_path = os.path.join(self.current_path, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"File deleted: {self.get_relative_path(file_path)}")
                self.update_context()
            except Exception as e:
                logging.error(f"Error deleting file {filename}: {e}")
        else:
            logging.warning(f"File not found: {self.get_relative_path(file_path)}")

class BrainAIShell(cmd.Cmd):
    intro = "Welcome to BrainAI. Type 'help' for a list of commands or 'exit' to quit."
    prompt = '> '

    def __init__(self):
        super().__init__()
        self.ai = BrainAI()
        self.update_prompt()

    def update_prompt(self):
        self.prompt = f"{self.ai.get_relative_path(self.ai.current_path)}> "

    def do_cd(self, arg):
        """Change directory: cd <directory>"""
        self.ai.change_directory(arg)
        self.update_prompt()

    def do_ls(self, arg):
        """List contents of current directory"""
        self.ai.list_directory()

    def do_search(self, arg):
        """Search for files containing a query: search <query>"""
        results = self.ai.search(arg)
        logging.info(f"Search results for '{arg}':")
        for result in results:
            logging.info(f"- {result}")

    def do_create(self, arg):
        """Create a new file: create <filename> <content>"""
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            logging.warning("Usage: create <filename> <content>")
            return
        filename, content = parts
        self.ai.create_file(filename, content)

    def do_edit(self, arg):
        """Edit an existing file: edit <filename> <new_content>"""
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            logging.warning("Usage: edit <filename> <new_content>")
            return
        filename, new_content = parts
        self.ai.edit_file(filename, new_content)

    def do_delete(self, arg):
        """Delete a file: delete <filename>"""
        self.ai.delete_file(arg)

    def do_exit(self, arg):
        """Exit the program"""
        logging.info("Goodbye!")
        return True

    def default(self, line):
        """Handle any input not matched by other do_* methods"""
        response = self.ai.ask_question(line)
        print(response)

    def completedefault(self, text, line, begidx, endidx):
        """Provide tab completion for file and directory names"""
        return [f for f in os.listdir(self.ai.current_path) if f.startswith(text)]

if __name__ == "__main__":
    BrainAIShell().cmdloop()
