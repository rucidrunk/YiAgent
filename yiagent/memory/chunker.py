"""
Text chunker for memory ingestion.

Splits document text into overlapping chunks, each sized to
chunk_max_tokens with chunk_overlap_tokens overlap between adjacent chunks.
Uses paragraph/line boundary awareness to avoid mid-sentence breaks.

Token counting is incremental (not O(n²)) — each paragraph/sentence is
estimated once and accumulated, rather than re-estimating the full buffer
on every merge check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from yiagent.common.utils import estimate_tokens, contains_cjk


@dataclass
class TextChunk:
    text: str
    start_line: int
    end_line: int
    token_estimate: int = 0


class TextChunker:
    """Splits text into semantically-aware overlapping chunks."""

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks with incremental token accounting."""
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        chunks: List[TextChunk] = []
        current_text = ""
        current_tokens = 0  # incremental counter — avoids O(n²) re-estimation
        current_start = 0
        current_end = 0
        line_offset = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            para_lines = para.count("\n") + 1

            if current_tokens + para_tokens > self.max_tokens and current_text:
                chunks.append(TextChunk(
                    text=current_text.strip(),
                    start_line=current_start,
                    end_line=current_end,
                    token_estimate=current_tokens,
                ))
                overlap_text = self._extract_overlap(current_text)
                current_text = overlap_text
                current_tokens = estimate_tokens(overlap_text)
                current_start = max(current_start, current_end - 1)

            if not current_text:
                current_start = line_offset
            current_text += para
            current_tokens += para_tokens
            current_end = line_offset + para_lines

            if para_tokens > self.max_tokens:
                sub_chunks = self._split_long_paragraph(para, current_start)
                if current_text == para:
                    chunks.extend(sub_chunks)
                    if sub_chunks:
                        last = sub_chunks[-1]
                        current_text = self._extract_overlap(last.text)
                        current_tokens = estimate_tokens(current_text)
                        current_start = last.start_line
                else:
                    if current_text != para:
                        chunks.append(TextChunk(
                            text=current_text.strip(),
                            start_line=current_start,
                            end_line=current_end,
                            token_estimate=current_tokens,
                        ))
                    chunks.extend(sub_chunks)
                    if sub_chunks:
                        last = sub_chunks[-1]
                        current_text = self._extract_overlap(last.text)
                        current_tokens = estimate_tokens(current_text)
                        current_start = last.start_line

            line_offset += para_lines

        if current_text.strip():
            chunks.append(TextChunk(
                text=current_text.strip(),
                start_line=current_start,
                end_line=line_offset,
                token_estimate=current_tokens,
            ))

        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        parts = re.split(r"(\n\n)", text)
        paragraphs: List[str] = []
        buf = ""
        for part in parts:
            if part == "\n\n":
                if buf:
                    paragraphs.append(buf + "\n\n")
                    buf = ""
                else:
                    paragraphs.append("\n\n")
            else:
                buf += part
        if buf:
            paragraphs.append(buf)
        return paragraphs or [text]

    def _split_long_paragraph(self, text: str, start_line: int) -> List[TextChunk]:
        sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
        chunks: List[TextChunk] = []
        current = ""
        cur_tokens = 0
        line_pos = start_line

        for sent in sentences:
            sent_tokens = estimate_tokens(sent)
            if cur_tokens + sent_tokens > self.max_tokens and current:
                chunks.append(TextChunk(
                    text=current.strip(),
                    start_line=line_pos,
                    end_line=line_pos + current.count("\n") + 1,
                    token_estimate=cur_tokens,
                ))
                line_pos += current.count("\n")
                overlap = self._extract_overlap(current)
                current = overlap + sent
                cur_tokens = estimate_tokens(overlap) + sent_tokens
            else:
                current += sent
                cur_tokens += sent_tokens

        if current.strip():
            chunks.append(TextChunk(
                text=current.strip(),
                start_line=line_pos,
                end_line=line_pos + current.count("\n") + 1,
                token_estimate=cur_tokens,
            ))
        return chunks

    def _extract_overlap(self, text: str) -> str:
        """Return the last ~overlap_tokens worth of text.

        Uses character-level slicing (not word splitting) so CJK text
        (which has no spaces) is handled identically to ASCII.
        """
        if self.overlap_tokens <= 0:
            return ""
        # Rough char count: CJK ~1.5 tokens/char, ASCII ~0.25 tokens/char.
        # Use a conservative divisor so we never exceed the intended budget.
        divisor = 1.5 if contains_cjk(text) else 0.25
        char_overlap = max(1, int(self.overlap_tokens / divisor))
        if len(text) <= char_overlap:
            return text + "\n\n"
        return text[-char_overlap:] + "\n\n"
