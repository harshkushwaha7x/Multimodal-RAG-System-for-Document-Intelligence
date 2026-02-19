"""
Computer Vision Pipeline for Document Image Processing.
Handles image preprocessing, enhancement, and OCR extraction.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ..utils import get_logger, LoggerMixin

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """Container for OCR extraction results."""
    
    text: str
    confidence: float
    bounding_boxes: List[Dict] = field(default_factory=list)
    page_number: int = 1
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_boxes": self.bounding_boxes,
            "page_number": self.page_number,
            "metadata": self.metadata
        }


@dataclass
class ImageQuality:
    """Image quality assessment metrics."""
    
    blur_score: float
    contrast_score: float
    brightness_score: float
    noise_level: float
    is_acceptable: bool
    
    def to_dict(self) -> Dict:
        return {
            "blur_score": self.blur_score,
            "contrast_score": self.contrast_score,
            "brightness_score": self.brightness_score,
            "noise_level": self.noise_level,
            "is_acceptable": self.is_acceptable
        }


class ImagePreprocessor(LoggerMixin):
    """
    Image preprocessing for OCR optimization.
    
    Applies various enhancement techniques to improve OCR accuracy:
    - Grayscale conversion
    - Adaptive thresholding
    - Noise reduction
    - Deskewing
    - Contrast enhancement (CLAHE)
    """
    
    def __init__(
        self,
        target_dpi: int = 300,
        denoise_strength: int = 10,
        adaptive_threshold_block_size: int = 11,
        adaptive_threshold_c: int = 2
    ):
        self.target_dpi = target_dpi
        self.denoise_strength = denoise_strength
        self.adaptive_threshold_block_size = adaptive_threshold_block_size
        self.adaptive_threshold_c = adaptive_threshold_c
    
    def preprocess(
        self,
        image: np.ndarray,
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_threshold: bool = True
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image (BGR or grayscale)
            apply_deskew: Whether to apply deskewing
            apply_denoise: Whether to apply noise reduction
            apply_threshold: Whether to apply adaptive thresholding
            
        Returns:
            Preprocessed image
        """
        self.logger.debug("Starting image preprocessing pipeline")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        gray = self._apply_clahe(gray)
        
        # Denoise
        if apply_denoise:
            gray = self._denoise(gray)
        
        # Deskew
        if apply_deskew:
            gray = self._deskew(gray)
        
        # Adaptive thresholding
        if apply_threshold:
            gray = self._adaptive_threshold(gray)
        
        self.logger.debug("Preprocessing pipeline completed")
        return gray
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        return cv2.fastNlMeansDenoising(
            image, 
            None, 
            h=self.denoise_strength, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct image skew using Hough Transform.
        
        Detects lines in the image and calculates the median angle
        to determine the skew angle, then rotates to correct.
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi / 180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 45:  # Filter out vertical lines
                angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle to avoid outliers
        median_angle = np.median(angles)
        
        if abs(median_angle) < 0.5:  # Skip if angle is negligible
            return image
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        self.logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
        return rotated
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive Gaussian thresholding."""
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_threshold_block_size,
            self.adaptive_threshold_c
        )
    
    def assess_quality(self, image: np.ndarray) -> ImageQuality:
        """
        Assess image quality for OCR suitability.
        
        Args:
            image: Input image
            
        Returns:
            ImageQuality metrics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500, 1.0)  # Normalize
        
        # Contrast score
        contrast_score = gray.std() / 128  # Normalize to ~1.0
        
        # Brightness score (0.5 is optimal)
        brightness = gray.mean() / 255
        brightness_score = 1 - abs(brightness - 0.5) * 2
        
        # Noise level estimation
        noise_level = self._estimate_noise(gray)
        
        # Determine if quality is acceptable
        is_acceptable = (
            blur_score > 0.1 and 
            contrast_score > 0.2 and 
            brightness_score > 0.3
        )
        
        return ImageQuality(
            blur_score=round(blur_score, 3),
            contrast_score=round(contrast_score, 3),
            brightness_score=round(brightness_score, 3),
            noise_level=round(noise_level, 3),
            is_acceptable=is_acceptable
        )
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian method."""
        sigma = np.median(np.abs(cv2.Laplacian(image, cv2.CV_64F))) / 0.6745
        return min(sigma / 50, 1.0)  # Normalize


class OCREngine(LoggerMixin):
    """
    OCR engine using Tesseract for text extraction.
    
    Provides text extraction with confidence scoring,
    bounding box detection, and multi-language support.
    """
    
    def __init__(
        self,
        lang: str = "eng",
        config: str = "--oem 3 --psm 3",
        min_confidence: float = 0.0
    ):
        """
        Initialize OCR engine.
        
        Args:
            lang: Tesseract language code
            config: Tesseract configuration string
            min_confidence: Minimum confidence threshold for text
        """
        self.lang = lang
        self.config = config
        self.min_confidence = min_confidence
        
        # Import pytesseract here to handle missing installation gracefully
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.logger.info("Tesseract OCR initialized successfully")
        except ImportError:
            self.pytesseract = None
            self.logger.warning(
                "pytesseract not installed. OCR functionality will be limited."
            )
    
    def extract_text(
        self,
        image: np.ndarray,
        with_confidence: bool = True
    ) -> OCRResult:
        """
        Extract text from an image.
        
        Args:
            image: Preprocessed image
            with_confidence: Whether to include confidence scores
            
        Returns:
            OCRResult with extracted text and metadata
        """
        if self.pytesseract is None:
            self.logger.error("pytesseract not available")
            return OCRResult(text="", confidence=0.0)
        
        self.logger.debug("Starting OCR extraction")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        if with_confidence:
            # Get detailed data with confidence
            data = self.pytesseract.image_to_data(
                pil_image,
                lang=self.lang,
                config=self.config,
                output_type=self.pytesseract.Output.DICT
            )
            
            # Process results
            text_parts = []
            bounding_boxes = []
            confidences = []
            
            for i, word in enumerate(data['text']):
                conf = float(data['conf'][i])
                
                if conf > self.min_confidence and word.strip():
                    text_parts.append(word)
                    confidences.append(conf)
                    
                    bounding_boxes.append({
                        'text': word,
                        'confidence': conf,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'block_num': data['block_num'][i],
                        'line_num': data['line_num'][i]
                    })
            
            text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
        else:
            # Simple text extraction
            text = self.pytesseract.image_to_string(
                pil_image,
                lang=self.lang,
                config=self.config
            )
            avg_confidence = 0.0
            bounding_boxes = []
        
        # Clean up text
        text = self._clean_text(text)
        
        self.logger.debug(f"OCR extracted {len(text)} characters with {avg_confidence:.1f}% confidence")
        
        return OCRResult(
            text=text,
            confidence=round(avg_confidence, 2),
            bounding_boxes=bounding_boxes
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()


class CVPipeline(LoggerMixin):
    """
    Complete Computer Vision pipeline for document processing.
    
    Combines image preprocessing and OCR into a unified pipeline
    for processing scanned documents and images.
    """
    
    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        ocr_engine: Optional[OCREngine] = None
    ):
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.ocr_engine = ocr_engine or OCREngine()
    
    def process_image(
        self,
        image_path: Union[str, Path],
        preprocess: bool = True
    ) -> OCRResult:
        """
        Process a single image file through the CV pipeline.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply preprocessing
            
        Returns:
            OCRResult with extracted text
        """
        image_path = Path(image_path)
        self.logger.info(f"Processing image: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return OCRResult(text="", confidence=0.0)
        
        # Assess quality
        quality = self.preprocessor.assess_quality(image)
        self.logger.debug(f"Image quality: {quality.to_dict()}")
        
        # Preprocess
        if preprocess:
            processed = self.preprocessor.preprocess(image)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # OCR
        result = self.ocr_engine.extract_text(processed)
        result.metadata['source_file'] = str(image_path)
        result.metadata['quality'] = quality.to_dict()
        
        return result
    
    def process_images(
        self,
        image_paths: List[Union[str, Path]],
        preprocess: bool = True
    ) -> List[OCRResult]:
        """
        Process multiple images through the CV pipeline.
        
        Args:
            image_paths: List of image file paths
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of OCRResults
        """
        results = []
        for i, path in enumerate(image_paths, 1):
            result = self.process_image(path, preprocess)
            result.page_number = i
            results.append(result)
        return results
    
    def process_pdf_images(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        preprocess: bool = True
    ) -> List[OCRResult]:
        """
        Convert PDF pages to images and process through OCR.
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for PDF to image conversion
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of OCRResults, one per page
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Processing PDF through OCR: {pdf_path.name}")
        
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(str(pdf_path), dpi=dpi)
            self.logger.debug(f"Converted PDF to {len(images)} images")
            
            results = []
            for i, pil_image in enumerate(images, 1):
                # Convert PIL to numpy
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Assess quality
                quality = self.preprocessor.assess_quality(image)
                
                # Preprocess
                if preprocess:
                    processed = self.preprocessor.preprocess(image)
                else:
                    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # OCR
                result = self.ocr_engine.extract_text(processed)
                result.page_number = i
                result.metadata['source_file'] = str(pdf_path)
                result.metadata['quality'] = quality.to_dict()
                
                results.append(result)
            
            return results
            
        except ImportError:
            self.logger.error("pdf2image not installed")
            return []
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return []


if __name__ == "__main__":
    # Test the CV pipeline
    import argparse
    
    parser = argparse.ArgumentParser(description="CV Pipeline Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--image", type=str, help="Image path to process")
    args = parser.parse_args()
    
    if args.test:
        print("CV Pipeline initialized successfully!")
        pipeline = CVPipeline()
        print(f"Preprocessor: {pipeline.preprocessor}")
        print(f"OCR Engine: {pipeline.ocr_engine}")
    
    if args.image:
        pipeline = CVPipeline()
        result = pipeline.process_image(args.image)
        print(f"\nExtracted Text:\n{result.text[:500]}...")
        print(f"\nConfidence: {result.confidence}%")
