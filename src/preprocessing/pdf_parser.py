"""
PDF Parser Module for Document Processing.
Extracts text, tables, and metadata from PDF documents.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..utils import get_logger, LoggerMixin

logger = get_logger(__name__)


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""
    
    page_number: int
    text: str
    tables: List[pd.DataFrame] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "tables": [t.to_dict() for t in self.tables],
            "metadata": self.metadata
        }


@dataclass
class DocumentContent:
    """Complete content extracted from a PDF document."""
    
    source_file: str
    total_pages: int
    pages: List[PageContent]
    metadata: Dict = field(default_factory=dict)
    
    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages)
    
    @property
    def all_tables(self) -> List[pd.DataFrame]:
        """Get all tables from all pages."""
        tables = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables
    
    def to_dict(self) -> Dict:
        return {
            "source_file": self.source_file,
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata
        }


class TableExtractor(LoggerMixin):
    """
    Extract tables from PDF documents.
    
    Uses Tabula and Camelot for table detection and extraction.
    """
    
    def __init__(self, method: str = "tabula"):
        """
        Initialize table extractor.
        
        Args:
            method: Extraction method - "tabula" or "camelot"
        """
        self.method = method
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required libraries are available."""
        if self.method == "tabula":
            try:
                import tabula
                self.tabula = tabula
                self.logger.debug("Tabula initialized successfully")
            except ImportError:
                self.tabula = None
                self.logger.warning("tabula-py not installed")
        elif self.method == "camelot":
            try:
                import camelot
                self.camelot = camelot
                self.logger.debug("Camelot initialized successfully")
            except ImportError:
                self.camelot = None
                self.logger.warning("camelot-py not installed")
    
    def extract_tables(
        self,
        pdf_path: Union[str, Path],
        pages: str = "all"
    ) -> Dict[int, List[pd.DataFrame]]:
        """
        Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to extract - "all" or specific pages like "1,2,3"
            
        Returns:
            Dictionary mapping page numbers to list of DataFrames
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Extracting tables from: {pdf_path.name}")
        
        if self.method == "tabula" and self.tabula:
            return self._extract_with_tabula(pdf_path, pages)
        elif self.method == "camelot" and self.camelot:
            return self._extract_with_camelot(pdf_path, pages)
        else:
            self.logger.warning("No table extraction library available")
            return {}
    
    def _extract_with_tabula(
        self,
        pdf_path: Path,
        pages: str
    ) -> Dict[int, List[pd.DataFrame]]:
        """Extract tables using Tabula."""
        try:
            # Read all tables
            tables = self.tabula.read_pdf(
                str(pdf_path),
                pages=pages,
                multiple_tables=True,
                pandas_options={'header': None}
            )
            
            # Group by page (tabula returns flat list)
            # For simplicity, assume sequential pages
            result = {}
            for i, table in enumerate(tables):
                if not table.empty:
                    page_num = i + 1
                    if page_num not in result:
                        result[page_num] = []
                    result[page_num].append(table)
            
            self.logger.debug(f"Extracted {len(tables)} tables")
            return result
            
        except Exception as e:
            self.logger.error(f"Tabula extraction failed: {e}")
            return {}
    
    def _extract_with_camelot(
        self,
        pdf_path: Path,
        pages: str
    ) -> Dict[int, List[pd.DataFrame]]:
        """Extract tables using Camelot."""
        try:
            tables = self.camelot.read_pdf(
                str(pdf_path),
                pages=pages if pages != "all" else "1-end",
                flavor='lattice'
            )
            
            result = {}
            for table in tables:
                page_num = table.page
                if page_num not in result:
                    result[page_num] = []
                result[page_num].append(table.df)
            
            self.logger.debug(f"Extracted {len(tables)} tables")
            return result
            
        except Exception as e:
            self.logger.error(f"Camelot extraction failed: {e}")
            return {}
    
    def table_to_text(self, table: pd.DataFrame) -> str:
        """
        Convert a DataFrame table to structured text.
        
        Args:
            table: Pandas DataFrame
            
        Returns:
            Formatted text representation
        """
        if table.empty:
            return ""
        
        # Clean up the table
        table = table.fillna("")
        
        # Convert to markdown-like format
        lines = []
        
        # Header (first row if it looks like headers)
        if len(table) > 1:
            header = " | ".join(str(cell) for cell in table.iloc[0])
            lines.append(header)
            lines.append("-" * len(header))
            start_row = 1
        else:
            start_row = 0
        
        # Data rows
        for _, row in table.iloc[start_row:].iterrows():
            line = " | ".join(str(cell) for cell in row)
            lines.append(line)
        
        return "\n".join(lines)


class ImageExtractor(LoggerMixin):
    """
    Extract images from PDF documents with optional captioning.
    
    Uses PyMuPDF for extraction and BLIP/CLIP for image understanding.
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        min_size: int = 100,
        generate_captions: bool = True
    ):
        """
        Initialize image extractor.
        
        Args:
            output_dir: Directory to save extracted images
            min_size: Minimum image dimension to extract
            generate_captions: Whether to generate captions using vision model
        """
        self.output_dir = Path(output_dir) if output_dir else Path("extracted_images")
        self.min_size = min_size
        self.generate_captions = generate_captions
        self.caption_model = None
        self.caption_processor = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_caption_model(self):
        """Lazy load BLIP captioning model."""
        if self.caption_model is None and self.generate_captions:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                import torch
                
                self.logger.info("Loading BLIP captioning model...")
                self.caption_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                
                if torch.cuda.is_available():
                    self.caption_model = self.caption_model.cuda()
                    
                self.logger.info("BLIP model loaded successfully")
            except Exception as e:
                self.logger.warning(f"BLIP not available: {e}")
                self.generate_captions = False
    
    def extract_images(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Extract images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract from (1-indexed), None for all
            
        Returns:
            List of dicts with image info: path, page, caption, etc.
        """
        try:
            import fitz
        except ImportError:
            self.logger.error("PyMuPDF required for image extraction")
            return []
        
        pdf_path = Path(pdf_path)
        self.logger.info(f"Extracting images from: {pdf_path.name}")
        
        extracted = []
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            if pages and (page_num + 1) not in pages:
                continue
                
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        
                        # Skip small images (likely icons/bullets)
                        if width < self.min_size or height < self.min_size:
                            continue
                        
                        # Save image
                        image_name = f"{pdf_path.stem}_p{page_num+1}_img{img_index+1}.{image_ext}"
                        image_path = self.output_dir / image_name
                        
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Generate caption if enabled
                        caption = ""
                        if self.generate_captions:
                            caption = self._generate_caption(image_path)
                        
                        extracted.append({
                            "path": str(image_path),
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "width": width,
                            "height": height,
                            "caption": caption,
                            "format": image_ext
                        })
                        
                except Exception as e:
                    self.logger.debug(f"Failed to extract image: {e}")
                    continue
        
        doc.close()
        self.logger.info(f"Extracted {len(extracted)} images")
        return extracted
    
    def _generate_caption(self, image_path: Path) -> str:
        """Generate caption for an image using BLIP."""
        self._load_caption_model()
        
        if not self.caption_model:
            return ""
        
        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert("RGB")
            inputs = self.caption_processor(images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.caption_model.generate(**inputs, max_new_tokens=50)
            
            caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            self.logger.debug(f"Caption generation failed: {e}")
            return ""
    
    def image_to_text(self, image_info: Dict) -> str:
        """Convert image info to text for indexing."""
        parts = [f"[Image on page {image_info['page']}]"]
        if image_info.get("caption"):
            parts.append(f"Caption: {image_info['caption']}")
        parts.append(f"Size: {image_info.get('width', '?')}x{image_info.get('height', '?')}")
        return " ".join(parts)


class OCRProcessor(LoggerMixin):
    """
    OCR processor for scanned PDFs.
    
    Uses Tesseract for text extraction from images.
    """
    
    def __init__(self, language: str = "eng"):
        """
        Initialize OCR processor.
        
        Args:
            language: Tesseract language code
        """
        self.language = language
        self.tesseract_available = self._check_tesseract()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.logger.info("Tesseract OCR available")
            return True
        except Exception:
            self.logger.warning("Tesseract not available. Install with: pip install pytesseract")
            return False
    
    def ocr_pdf(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 300
    ) -> List[Dict]:
        """
        Perform OCR on a scanned PDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering pages
            
        Returns:
            List of dicts with page number and extracted text
        """
        if not self.tesseract_available:
            return []
        
        try:
            import fitz
            import pytesseract
            from PIL import Image
            import io
        except ImportError as e:
            self.logger.error(f"Missing dependency: {e}")
            return []
        
        pdf_path = Path(pdf_path)
        self.logger.info(f"OCR processing: {pdf_path.name}")
        
        results = []
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.language)
            
            results.append({
                "page": page_num + 1,
                "text": text.strip(),
                "confidence": self._get_confidence(image)
            })
        
        doc.close()
        self.logger.info(f"OCR completed for {len(results)} pages")
        return results
    
    def _get_confidence(self, image) -> float:
        """Get OCR confidence score."""
        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data.get("conf", []) if c != "-1" and int(c) > 0]
            return sum(confidences) / len(confidences) if confidences else 0.0
        except:
            return 0.0



class PDFParser(LoggerMixin):
    """
    PDF Parser for extracting text and structure from documents.
    
    Uses PyMuPDF (fitz) for fast and accurate text extraction
    with layout preservation.
    """
    
    def __init__(
        self,
        extract_tables: bool = True,
        table_extractor: Optional[TableExtractor] = None
    ):
        """
        Initialize PDF parser.
        
        Args:
            extract_tables: Whether to extract tables
            table_extractor: Custom table extractor instance
        """
        self.extract_tables = extract_tables
        self.table_extractor = table_extractor or TableExtractor()
        
        # Import fitz (PyMuPDF)
        try:
            import fitz
            self.fitz = fitz
            self.logger.info("PyMuPDF initialized successfully")
        except ImportError:
            self.fitz = None
            self.logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
    
    def parse(
        self,
        pdf_path: Union[str, Path],
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> DocumentContent:
        """
        Parse a PDF document and extract all content.
        
        Args:
            pdf_path: Path to PDF file
            start_page: First page to process (0-indexed)
            end_page: Last page to process (exclusive, None for all)
            
        Returns:
            DocumentContent with extracted text and tables
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Parsing PDF: {pdf_path.name}")
        
        if self.fitz is None:
            self.logger.error("PyMuPDF not available")
            return DocumentContent(
                source_file=str(pdf_path),
                total_pages=0,
                pages=[],
                metadata={}
            )
        
        try:
            doc = self.fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            # Determine page range
            if end_page is None:
                end_page = total_pages
            end_page = min(end_page, total_pages)
            
            # Extract document metadata
            doc_metadata = self._extract_metadata(doc)
            
            # Extract tables if enabled
            tables_by_page = {}
            if self.extract_tables:
                page_spec = f"{start_page + 1}-{end_page}"
                tables_by_page = self.table_extractor.extract_tables(pdf_path, page_spec)
            
            # Process each page
            pages = []
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text = self._extract_page_text(page)
                
                # Get tables for this page
                page_tables = tables_by_page.get(page_num + 1, [])
                
                # Page metadata
                page_metadata = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                }
                
                page_content = PageContent(
                    page_number=page_num + 1,
                    text=text,
                    tables=page_tables,
                    metadata=page_metadata
                )
                pages.append(page_content)
            
            doc.close()
            
            self.logger.info(f"Extracted {len(pages)} pages from PDF")
            
            return DocumentContent(
                source_file=str(pdf_path),
                total_pages=total_pages,
                pages=pages,
                metadata=doc_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF: {e}")
            return DocumentContent(
                source_file=str(pdf_path),
                total_pages=0,
                pages=[],
                metadata={"error": str(e)}
            )
    
    def _extract_metadata(self, doc) -> Dict:
        """Extract document metadata."""
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "keywords": metadata.get("keywords", "")
        }
    
    def _extract_page_text(self, page) -> str:
        """
        Extract text from a page with layout preservation.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        # Get text blocks for layout-aware extraction
        blocks = page.get_text("blocks")
        
        # Sort blocks by position (top to bottom, left to right)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        
        # Extract text from blocks
        text_parts = []
        for block in blocks:
            if block[6] == 0:  # Text block (not image)
                text = block[4].strip()
                if text:
                    text_parts.append(text)
        
        text = "\n\n".join(text_parts)
        
        # Clean up text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Fix hyphenation at line breaks
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    def extract_sections(self, doc_content: DocumentContent) -> List[Dict]:
        """
        Extract sections/headings from document.
        
        Args:
            doc_content: Parsed document content
            
        Returns:
            List of sections with titles and content
        """
        sections = []
        current_section = {"title": "Introduction", "content": [], "page": 1}
        
        # Simple heuristic: lines that are short and possibly uppercase are headings
        for page in doc_content.pages:
            lines = page.text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line looks like a heading
                is_heading = (
                    len(line) < 100 and
                    (line.isupper() or 
                     line.istitle() or
                     re.match(r'^\d+\.?\s+[A-Z]', line))
                )
                
                if is_heading:
                    # Save current section
                    if current_section["content"]:
                        current_section["content"] = "\n".join(current_section["content"])
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": line,
                        "content": [],
                        "page": page.page_number
                    }
                else:
                    current_section["content"].append(line)
        
        # Don't forget last section
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)
        
        return sections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Parser Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--pdf", type=str, help="PDF path to process")
    args = parser.parse_args()
    
    if args.test:
        print("PDF Parser initialized successfully!")
        pdf_parser = PDFParser()
        print(f"PyMuPDF available: {pdf_parser.fitz is not None}")
        print(f"Table extraction enabled: {pdf_parser.extract_tables}")
    
    if args.pdf:
        pdf_parser = PDFParser()
        result = pdf_parser.parse(args.pdf)
        print(f"\nDocument: {result.source_file}")
        print(f"Total pages: {result.total_pages}")
        print(f"Metadata: {result.metadata}")
        print(f"\nFirst page text (first 500 chars):\n{result.pages[0].text[:500]}...")
