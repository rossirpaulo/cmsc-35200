# External dependencies:
# pip install requests pyyaml openai
import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import requests
import yaml

# Base URL for ArXiv API
BASE_URL = "http://export.arxiv.org/api/query"

# Optional imports for PDF validation
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


@dataclass
class ValidationResult:
    """Result of PDF validation"""
    file_path: str
    file_size: int
    is_valid: bool
    is_parsable: bool
    validation_method: str
    error_message: Optional[str] = None
    validation_time: float = 0.0
    page_count: Optional[int] = None
    is_encrypted: bool = False
    is_scanned: Optional[bool] = None


class PDFValidator:
    """Comprehensive PDF validation using multiple methods with qpdf as last resort"""

    def __init__(self, use_qpdf_as_fallback: bool = True):
        self.use_qpdf_as_fallback = use_qpdf_as_fallback
        self.qpdf_available = self._check_qpdf_available()

        if not any([self.qpdf_available, HAS_PYPDF2, HAS_PYMUPDF]):
            print(
                "Warning: No PDF validation tools available. Install qpdf, PyPDF2, or PyMuPDF for validation."
            )

    def _check_qpdf_available(self) -> bool:
        """Check if qpdf is available"""
        try:
            subprocess.run(["qpdf", "--version"], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _basic_file_checks(self, pdf_path: str) -> Optional[ValidationResult]:
        """Perform basic file checks before PDF validation"""
        try:
            if not os.path.exists(pdf_path):
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=0,
                    is_valid=False,
                    is_parsable=False,
                    validation_method="file_check",
                    error_message="File not found",
                )

            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=0,
                    is_valid=False,
                    is_parsable=False,
                    validation_method="file_check",
                    error_message="Empty file",
                )

            if file_size < 100:  # PDFs have minimum size
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=file_size,
                    is_valid=False,
                    is_parsable=False,
                    validation_method="file_check",
                    error_message="File too small to be valid PDF",
                )

            # Check PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    return ValidationResult(
                        file_path=pdf_path,
                        file_size=file_size,
                        is_valid=False,
                        is_parsable=False,
                        validation_method="file_check",
                        error_message="Invalid PDF header",
                    )

            return None  # Passed basic checks

        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=0,
                is_valid=False,
                is_parsable=False,
                validation_method="file_check",
                error_message=str(e),
            )

    def _validate_with_pymupdf(self, pdf_path: str) -> ValidationResult:
        """Validate PDF using PyMuPDF (most practical method)"""
        start_time = time.time()

        try:
            doc = fitz.open(pdf_path)
            is_encrypted = doc.needs_pass
            page_count = len(doc)

            # Quick heuristic to check if it's a scanned PDF
            is_scanned = None
            if page_count > 0 and not is_encrypted:
                first_page = doc[0]
                text = first_page.get_text()
                is_scanned = len(text.strip()) < 10  # Very little text suggests scanned

            doc.close()
            validation_time = time.time() - start_time

            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=True,
                is_parsable=not is_encrypted and not is_scanned,
                validation_method="PyMuPDF",
                validation_time=validation_time,
                page_count=page_count,
                is_encrypted=is_encrypted,
                is_scanned=is_scanned,
            )

        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="PyMuPDF",
                error_message=str(e),
                validation_time=time.time() - start_time,
            )

    def _validate_with_pypdf2(self, pdf_path: str) -> ValidationResult:
        """Validate PDF using PyPDF2"""
        start_time = time.time()

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                is_encrypted = reader.is_encrypted
                page_count = len(reader.pages)

                # Quick validation by accessing first page
                if page_count > 0:
                    first_page = reader.pages[0]
                    _ = first_page.mediabox  # This validates basic structure

                validation_time = time.time() - start_time

                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=True,
                    is_parsable=not is_encrypted,
                    validation_method="PyPDF2",
                    validation_time=validation_time,
                    page_count=page_count,
                    is_encrypted=is_encrypted,
                )

        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="PyPDF2",
                error_message=str(e),
                validation_time=time.time() - start_time,
            )

    def _validate_with_qpdf(self, pdf_path: str) -> ValidationResult:
        """Validate PDF using qpdf (fallback method - strict but reliable)"""
        start_time = time.time()

        try:
            result = subprocess.run(
                ["qpdf", "--check", pdf_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            validation_time = time.time() - start_time

            if result.returncode == 0:
                # Get page count
                page_count = self._get_page_count_qpdf(pdf_path)

                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=True,
                    is_parsable=True,  # qpdf validation implies parsability
                    validation_method="qpdf",
                    validation_time=validation_time,
                    page_count=page_count,
                )
            else:
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=False,
                    is_parsable=False,
                    validation_method="qpdf",
                    error_message=f"qpdf strict validation failed. Note: file may still work in viewers. Error: {result.stderr.strip()}",
                    validation_time=validation_time,
                )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="qpdf",
                error_message="Validation timeout",
                validation_time=time.time() - start_time,
            )
        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="qpdf",
                error_message=str(e),
                validation_time=time.time() - start_time,
            )

    def _get_page_count_qpdf(self, pdf_path: str) -> Optional[int]:
        """Get page count using qpdf"""
        try:
            result = subprocess.run(
                ["qpdf", "--show-npages", pdf_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return None

    def validate(self, pdf_path: str) -> ValidationResult:
        """
        Validate a PDF file using the best available method.
        Priority order: PyMuPDF > PyPDF2 > qpdf (as last resort)
        """
        # Basic file checks first
        basic_result = self._basic_file_checks(pdf_path)
        if basic_result:
            return basic_result

        # Try PyMuPDF first (most practical - matches PDF viewer behavior)
        if HAS_PYMUPDF:
            result = self._validate_with_pymupdf(pdf_path)
            if result.is_valid:
                return result
            print(f"PyMuPDF validation failed for {pdf_path}: {result.error_message}")

        # Try PyPDF2 as backup
        if HAS_PYPDF2:
            result = self._validate_with_pypdf2(pdf_path)
            if result.is_valid:
                return result
            print(f"PyPDF2 validation failed for {pdf_path}: {result.error_message}")

        # Use qpdf as last resort if enabled and available
        if self.use_qpdf_as_fallback and self.qpdf_available:
            print(f"Using qpdf as last resort for {pdf_path}")
            result = self._validate_with_qpdf(pdf_path)
            if not result.is_valid:
                # Add context that qpdf is strict
                result.error_message = f"qpdf validation failed: {result.error_message}. Note: qpdf is strict - file may still work in viewers."
            return result

        # No validation method succeeded or available
        return ValidationResult(
            file_path=pdf_path,
            file_size=os.path.getsize(pdf_path),
            is_valid=False,
            is_parsable=False,
            validation_method="none",
            error_message="No validation method available or all methods failed",
        )


def sanitize_filename(name):
    """Create a file-system friendly name."""
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_")).rstrip()


def search_papers(keyword, start=0, max_results=100):
    """
    Search papers for a given 'keyword' using the ArXiv API.
    Returns a list of dictionaries representing the papers.
    """
    query = urllib.parse.quote(keyword)
    url = f"{BASE_URL}?search_query=all:{query}&start={start}&max_results={max_results}"
    
    print("-" * 50)
    print(f"Debug Info: Attempting to retrieve papers from ArXiv with:")
    print(f"  Keyword: {keyword}")
    print(f"  URL: {url}")
    print("-" * 50)

    try:
        response = requests.get(url, timeout=30)
        print(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            # ArXiv uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                paper_id = entry.find('atom:id', ns).text.split('/')[-1]
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip()
                published = entry.find('atom:published', ns).text
                year = published.split('-')[0]
                
                # Find PDF link
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break
                
                if not pdf_url:
                    # Fallback: construct PDF URL from ID if not explicit
                    # Usually http://arxiv.org/pdf/{id}
                    pdf_url = f"http://arxiv.org/pdf/{paper_id}.pdf"

                paper = {
                    "paperId": paper_id,
                    "title": title,
                    "abstract": summary,
                    "year": year,
                    "openAccessPdf": {"url": pdf_url} if pdf_url else None
                }
                papers.append(paper)
            
            print(f"Successfully parsed {len(papers)} papers from ArXiv response.")
            return {"data": papers, "next": len(papers) == max_results}
            
        else:
            print(f"Error: Unexpected status code {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Network or request error: {e}")
        return None
    except ET.ParseError as e:
        print(f"XML Parsing error: {e}")
        return None


def validate_pdf_comprehensive(pdf_path):
    """
    Validate a PDF file using comprehensive validation methods.
    """
    validator = PDFValidator(use_qpdf_as_fallback=True)
    result = validator.validate(pdf_path)

    if result.is_valid and result.is_parsable:
        print(f"PDF validation successful ({result.validation_method}): {pdf_path}")
        if result.page_count:
            print(f"  Pages: {result.page_count}")
        return True
    else:
        print(f"PDF validation failed ({result.validation_method}): {pdf_path}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        return False


def download_pdf(pdf_url, save_path):
    """
    Download a PDF from 'pdf_url' and save it to 'save_path'.
    """
    print(f"Attempting to download PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url, stream=True, timeout=60)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            print(f"Downloaded PDF to temporary file: {temp_path}")

            if validate_pdf_comprehensive(temp_path):
                import shutil
                shutil.move(temp_path, save_path)
                print(f"PDF validated and saved to: {save_path}")
                return True
            else:
                os.unlink(temp_path)
                print(f"Invalid PDF downloaded from {pdf_url}. File not saved.")
                return False
        else:
            print(f"Failed to download PDF from {pdf_url} - status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception while downloading PDF from {pdf_url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download academic papers from ArXiv.")
    parser.add_argument("--keywords", type=str, help="Keywords to search for.")
    parser.add_argument("--master-dir", type=str, default=None, help="Directory to save papers.")
    parser.add_argument("--limit", type=int, default=200, help="Total papers to download.")

    args = parser.parse_args()
    
    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else []
    if not keywords:
        print("No keywords provided.")
        sys.exit(1)

    for keyword in keywords:
        print("=" * 50)
        print(f"Processing keyword: '{keyword}'")
        print("=" * 50)

        directory_name = sanitize_filename(keyword)
        if args.master_dir:
            os.makedirs(args.master_dir, exist_ok=True)
            directory_name = os.path.join(args.master_dir, directory_name)
        os.makedirs(directory_name, exist_ok=True)

        processed_papers_file = os.path.join(directory_name, "processed_papers.txt")
        processed_papers = set()
        if os.path.exists(processed_papers_file):
            with open(processed_papers_file, "r", encoding="utf-8") as f:
                processed_papers = set(line.strip() for line in f)

        all_papers = []
        total_fetched = 0
        batch_size = 50
        
        while total_fetched < args.limit:
            print(f"Fetching papers (start={total_fetched})...")
            data = search_papers(keyword, start=total_fetched, max_results=batch_size)
            
            if data and "data" in data:
                papers = data["data"]
                if not papers:
                    break
                all_papers.extend(papers)
                total_fetched += len(papers)
                print(f"Retrieved {len(papers)} papers. Total: {total_fetched}")
                
                if not data.get("next") or len(papers) < batch_size:
                    break
            else:
                break
            
            time.sleep(3) # Be nice to ArXiv API

        print(f"Total papers collected for '{keyword}': {len(all_papers)}")
        
        pdfs_downloaded = 0
        
        for paper in all_papers:
            if pdfs_downloaded >= args.limit:
                break
                
            paper_id = paper["paperId"]
            title = paper["title"]
            
            if paper_id in processed_papers:
                continue

            print(f"\nProcessing Paper ID: {paper_id} - '{title}'")
            
            # Save abstract
            abstract_filename = os.path.join(directory_name, f"{paper_id}.txt")
            with open(abstract_filename, "w", encoding="utf-8") as f:
                f.write(f"Title: {title}\n")
                f.write(f"Year: {paper['year']}\n")
                f.write(f"Paper ID: {paper_id}\n")
                f.write(f"Abstract:\n{paper['abstract']}\n")

            # Download PDF
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"]["url"]
                pdf_filename = f"{paper_id}.pdf"
                save_path = os.path.join(directory_name, pdf_filename)
                
                if not os.path.exists(save_path):
                    if download_pdf(pdf_url, save_path):
                        pdfs_downloaded += 1
                else:
                    print(f"PDF already exists: {save_path}")
                    pdfs_downloaded += 1
            
            processed_papers.add(paper_id)
            with open(processed_papers_file, "a", encoding="utf-8") as f:
                f.write(paper_id + "\n")

        print(f"\nFinished processing for keyword '{keyword}'.")
        print(f"  PDFs downloaded/verified: {pdfs_downloaded}")

if __name__ == "__main__":
    main()

