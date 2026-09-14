from rag_qa.edu_document_loaders.edu_docloader import OCRDOCLoader
from rag_qa.edu_document_loaders.edu_pptloader import OCRPPTLoader
from rag_qa.edu_document_loaders.edu_imgloader import OCRIMGLoader
from rag_qa.edu_document_loaders.edu_pdfloader import OCRPDFLoader
from rag_qa.edu_document_loaders.edu_ocr import get_ocr

__all__ = ["OCRDOCLoader", "OCRPPTLoader", "OCRIMGLoader", "OCRPDFLoader", "get_ocr"]
