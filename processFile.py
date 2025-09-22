from typing import List, Dict, Any, Optional
import json
import os

class ProcessFile():
  def __init__(self, filepath):
    self.filepath = filepath
    self.file_path = filepath
    self.folder_path = filepath

  def load_document(self, file_path)-> list: 
    content = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
      for line in f:
        content.append(line)
        # return f.read()
    return content

  def process_txt_file(self, folder_path:str, filename:str, documents:list):
    file_path = os.path.join(folder_path, filename)
    print(f"\nProcessing document: {filename}")

    try:
        content = self.load_document(file_path)
        # text = quitar_acentos_regex(text)
        for text in content:
          doc_info = {
              "filename": filename,
              "path": file_path,
              "content": text,
              "length_chars": len(text),
              "length_words": len(text.split())
          }
          documents.append(doc_info)
        print(f"✓ Loaded {doc_info['length_chars']} chars, {doc_info['length_words']} words")
    except Exception as e:
        print(f"⚠ Skipped {filename} (Error: {e})")

  def process_jsonl_file(self, folder_path:str, filename:str, documents:list):
    file_path = os.path.join(folder_path, filename)
    print(f"\nProcessing document: {filename}")
    try:
        content = self.load_document(file_path)
        # text = quitar_acentos_regex(text)
        for line in content:
          line_content = json.loads(line)
          text = line_content['context']

          doc_info = {
              "filename": filename,
              "path": file_path,
              "content": text,
              "length_chars": len(text),
              "length_words": len(text.split())
          }
          documents.append(doc_info)
          print(f"✓ Loaded {doc_info['length_chars']} chars, {doc_info['length_words']} words")
    except Exception as e:
        print(f"⚠ Skipped {filename} (Error: {e})")
    
  def load_texts(self) -> List[Dict]:
      documents: List[Dict[str, Any]] = []
      print(f"Scanning folder: {self.folder_path}")

      for filename in os.listdir(self.folder_path):
          print('filename to scan', filename)
          if filename.endswith(".txt"):
            self.process_txt_file(self.folder_path, filename, documents)
          if filename.endswith(".jsonl"):
            self.process_jsonl_file(self.folder_path, filename, documents)
              

      print(f"\nTotal documents loaded: {len(documents)}")
      return documents
