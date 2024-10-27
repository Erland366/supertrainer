__all__ = ["DEFAULT_INSTRUCTION_TEMPLATE", "DEFAULT_INPUT_TEMPLATE"]

DEFAULT_INSTRUCTION_TEMPLATE = "Tentukan sentimen (positif, netral, atau negatif) pada teks berikut dari sudut pandang {entity}"
DEFAULT_INPUT_TEMPLATE = "\n[Judul]: {title}\n[/Judul]\n[Konten]: {content}\n[/Konten]"
