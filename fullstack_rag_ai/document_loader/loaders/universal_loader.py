import os
#import fitz
#from uuid import uuid4
from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader,
    Docx2txtLoader
)
#from .vision_parser import VisionParser


class UniversalLoader:

    def __init__(self, multimodal: bool = False):
        self.multimodal = multimodal
        # if multimodal:
        #     self.vision_parser = VisionParser()

        #     self.image_dir = "tmp_images"

        #     os.makedirs(
        #         self.image_dir,
        #         exist_ok=True,
        #     )

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()

        try:
            # if ext.lower() == ".pdf" and self.multimodal:
            #     docs = self._load_multimodal_pdf(file_path)
            
            if ext.lower() == ".pdf":
                docs = PyPDFLoader(file_path).load()
            elif ext.lower() == ".csv":
                docs = CSVLoader(file_path).load()

            elif ext.lower() in [".doc", ".docx"]:
                docs = Docx2txtLoader(file_path).load()

            elif ext.lower() in [
                ".txt", ".md", ".py", ".js",
                ".ts", ".html", ".css",
                ".json", ".yaml", ".yml"
            ]:
                docs = TextLoader(file_path, encoding="utf-8").load()

            else:
                docs = UnstructuredFileLoader(file_path).load()    
            for d in docs:
                d.metadata["source"] = file_path
                d.metadata["file_type"] = ext

            return docs

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            return []

    # ========================================================
    # MULTIMODAL PDF LOADER
    # ========================================================

#     def _load_multimodal_pdf(self, pdf_path: str) -> List[Document]:

#         pdf = fitz.open(pdf_path)

#         merged_text = ""

#         for page_number, page in enumerate(pdf):

#             blocks = page.get_text("dict")["blocks"]

#             # preserve reading order
#             blocks = sorted(
#                 blocks,
#                 key=lambda b: (b["bbox"][1], b["bbox"][0])
#             )

#             for block in blocks:

#                 # =========================
#                 # TEXT BLOCK
#                 # =========================
#                 if block["type"] == 0:

#                     text = self._extract_text_block(block)

#                     if text.strip():
#                         merged_text += text + "\n\n"

#                 # =========================
#                 # IMAGE BLOCK
#                 # =========================
#                 elif block["type"] == 1:

#                     try:

#                         image_path = self._extract_image(
#                             pdf=pdf,
#                             page=page,
#                             block=block,
#                             page_number=page_number
#                         )

#                         if not image_path:
#                             continue

#                         analysis = self.vision_parser.analyze_image(image_path)

#                         merged_text += f"""
# [IMAGE_ANALYSIS]

# Page: {page_number + 1}

# {analysis}

# [/IMAGE_ANALYSIS]

# """

#                     except Exception as e:
#                         print(f"[WARNING] Image parsing failed: {e}")

#         return [
#             Document(
#                 page_content=merged_text,
#                 metadata={
#                     "source": pdf_path,
#                     "type": "multimodal_pdf",
#                 },
#             )
#         ]

#     # ========================================================
#     # TEXT EXTRACTION
#     # ========================================================

#     def _extract_text_block(self, block) -> str:

#         text = ""

#         for line in block.get("lines", []):
#             for span in line.get("spans", []):
#                 text += span.get("text", "") + " "

#         return text.strip()

#     # ========================================================
#     # IMAGE EXTRACTION (FIXED HYBRID)
#     # ========================================================

#     def _extract_image(
#         self,
#         pdf,
#         page,
#         block,
#         page_number,
#     ) -> str:

#         image_path = None

#         # ----------------------------------------------------
#         # CASE 1: embedded image (xref exists)
#         # ----------------------------------------------------
#         xref = block.get("xref", None)

#         if xref:

#             try:
#                 pix = fitz.Pixmap(pdf, xref)

#                 image_path = self._save_pixmap(pix, page_number)

#                 return image_path

#             except Exception:
#                 pass  # fallback to bbox

#         # ----------------------------------------------------
#         # CASE 2: fallback → render bbox (MOST IMPORTANT)
#         # ----------------------------------------------------
#         bbox = block.get("bbox", None)

#         if bbox:

#             try:

#                 rect = fitz.Rect(bbox)

#                 mat = fitz.Matrix(2, 2)

#                 pix = page.get_pixmap(
#                     matrix=mat,
#                     clip=rect,
#                     alpha=False,
#                 )

#                 image_path = self._save_pixmap(pix, page_number)

#                 return image_path

#             except Exception as e:
#                 print(f"[WARNING] bbox rendering failed: {e}")

#         return None

#     # ========================================================
#     # SAVE PIXMAP
#     # ========================================================

#     def _save_pixmap(self, pix, page_number):

#         try:

#             if pix.alpha:
#                 pix = fitz.Pixmap(fitz.csRGB, pix)

#             image_name = f"{uuid4().hex}_page_{page_number}.png"

#             image_path = os.path.join(self.image_dir, image_name)

#             pix.save(image_path)

#             return image_path

#         except Exception as e:
#             print(f"[WARNING] Failed saving image: {e}")
#             return None