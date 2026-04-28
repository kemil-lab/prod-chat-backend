from typing import List


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
  
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap,1)

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks



import re
from typing import List


def split_into_paragraphs(text: str) -> List[str]:

    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_paragraphs = re.split(r"\n\s*\n+", text)

    paragraphs = []
    for para in raw_paragraphs:
        cleaned = " ".join(para.split()).strip()
        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


def paragraph_chunk_text(
    text: str,
    max_words: int = 250,
    overlap_words: int = 40,
) -> List[str]:
 
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []
    current_chunk_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > max_words:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []

            start = 0
            step = max(max_words - overlap_words, 1)

            while start < len(para_words):
                end = start + max_words
                piece = para_words[start:end]
                chunks.append(" ".join(piece))
                start += step

            continue

        if len(current_chunk_words) + len(para_words) <= max_words:
            current_chunk_words.extend(para_words)
        else:
            chunks.append(" ".join(current_chunk_words))

            overlap = current_chunk_words[-overlap_words:] if overlap_words > 0 else []

            current_chunk_words = overlap + para_words

            if len(current_chunk_words) > max_words:
                current_chunk_words = current_chunk_words[-max_words:]

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks