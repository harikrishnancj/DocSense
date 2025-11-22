from __future__ import annotations

from typing import List

from llama_index.core import Document

from states.loaders.utils import client


def load_audio(path: str, filename: str) -> List[Document]:
    transcription = ""
    if client is not None:
        try:
            with open(path, "rb") as fh:
                resp = client.audio.transcriptions.create(file=fh, model="whisper-1")  # type: ignore[attr-defined]
                transcription = getattr(resp, "text", "") or ""
        except Exception:
            transcription = ""
    return [
        Document(
            text=f"[AUDIO]\nFilename:{filename}\nTranscription:{transcription or '[NO TRANSCRIPTION]'}",
            metadata={"filename": filename, "type": "audio"},
        )
    ]

