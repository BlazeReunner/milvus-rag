from pathlib import Path
from typing import List, Dict, Optional
import os
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    print("Warning: pdfplumber not installed. PDF loading will not work. Install with: pip install pdfplumber")

try:
    import docx
except ImportError:
    docx = None
    print("Warning: python-docx not installed. DOCX loading will not work. Install with: pip install python-docx")


def load_pdf(path: Path) -> str:
    """
    Load text content from a PDF file.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Extracted text from all pages of the PDF
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber is required to load PDF files. Install with: pip install pdfplumber")
    
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        raise Exception(f"Error reading PDF file {path}: {str(e)}")
    
    return "\n".join(text)


def load_docx(path: Path) -> str:
    """
    Load text content from a DOCX file.
    
    Args:
        path: Path to the DOCX file
        
    Returns:
        Extracted text from all paragraphs in the document
    """
    if docx is None:
        raise ImportError("python-docx is required to load DOCX files. Install with: pip install python-docx")
    
    try:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        raise Exception(f"Error reading DOCX file {path}: {str(e)}")


def load_text(path: Path) -> str:
    """
    Load text content from a TXT or MD file.
    
    Args:
        path: Path to the text file
        
    Returns:
        Content of the text file
    """
    try:
        # Try UTF-8 first, fall back to other encodings if needed
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try latin-1 as fallback
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            raise Exception(f"Error reading text file {path}: {str(e)}")


def get_file_metadata(path: Path) -> Dict:
    """
    Extract metadata from a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary containing file metadata
    """
    stat = path.stat()
    
    metadata = {
        "source": path.name,
        "file_path": str(path),
        "file_extension": path.suffix.lower(),
        "file_size": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }
    
    # Add page count for PDFs if available
    if path.suffix.lower() == ".pdf" and pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                metadata["page_count"] = len(pdf.pages)
        except:
            pass
    
    return metadata


def load_file(path: Path) -> Optional[Dict]:
    """
    Load a single file and return its text content and metadata.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with 'text' and 'metadata' keys, or None if file type not supported
    """
    suffix = path.suffix.lower()
    
    try:
        if suffix == ".pdf":
            text = load_pdf(path)
        elif suffix == ".docx":
            text = load_docx(path)
        elif suffix in [".txt", ".md"]:
            text = load_text(path)
        else:
            return None
        
        # Clean text: remove excessive whitespace
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        metadata = get_file_metadata(path)
        metadata["text_length"] = len(text)
        metadata["word_count"] = len(text.split())
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    except Exception as e:
        print(f"Error loading file {path}: {str(e)}")
        return None


def load_all_docs(folder: str, recursive: bool = True) -> List[Dict]:
    """
    Load all supported documents from a folder.
    
    Args:
        folder: Path to the folder containing documents
        recursive: If True, search recursively in subdirectories
        
    Returns:
        List of dictionaries, each containing 'text' and 'metadata' keys
    """
    folder_path = Path(folder)
    
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")
    
    docs = []
    supported_extensions = {".pdf", ".docx", ".txt", ".md"}
    
    # Get all files matching supported extensions
    if recursive:
        files = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions]
    else:
        files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
    
    for file_path in files:
        result = load_file(file_path)
        if result:
            docs.append(result)
    
    return docs


def load_docs_from_paths(file_paths: List[str]) -> List[Dict]:
    """
    Load documents from a list of file paths.
    
    Args:
        file_paths: List of file paths as strings
        
    Returns:
        List of dictionaries, each containing 'text' and 'metadata' keys
    """
    docs = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File does not exist: {file_path}")
            continue
        
        result = load_file(path)
        if result:
            docs.append(result)
    
    return docs

