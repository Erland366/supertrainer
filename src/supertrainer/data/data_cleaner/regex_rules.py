# MIT License
#
# Copyright (c) 2024 Edd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re

import html2text
import yaml


class RegexConfig:
    BACA_SELENGKAPNYA_TAG = r"\b[B|b]aca selengkapnya *(\w| )*."

    # Find everything on the start of the text until it found -
    # The amount of words can be up to 4
    STARTING_TAG = r"^(?:\S+\s+){0,4}\S*-\s*"

    # Find entire line that has source in it
    SOURCE_TAG = r"\n\s*Source\s*:\s*.+"

    # Find M. Eng
    M_ENG_TAG = r"\s*M\.?\s*Eng"

    # Find PhD Tag
    PHD_TAG = r"\s*Ph\.?\s*D\."

    # Find entire line that has "Baca Juga" in it
    BACA_JUGA_TAG = r"\n\s*Baca juga\s*:\s*.+"

    # Find consecutive hastags (Hastag that's more than one without space)
    CONSECUTIVE_HASHTAG_TAG = r"(#\w+\s+)(#\w+\s*)+\s?"

    CONSECUTIVE_DOT_TAG = r"\.(\s*\.)*"

    # Find links
    LINKS_TAG = r"https?://\S+|www\.\S+|ftp://\S+"

    # Find new line
    NEW_LINE_TAG = r"\n"

    # Find Reuters tag
    REUTERS_TAG = r"<p>(?:(?!<\/?p>|Reuters).)*Reuters(?:(?!<\/?p>).)*<\/p>\n?"

    # Find everything that uses CAPITAL LETTER that's only one or two letter near dot.
    # Mainly used for academic title
    ONE_TWO_DOT_TAG = r"(\b\w{1,2})\."

    # Find Image markdown tag
    IMAGE_MARKDOWN_TAG = r"!\[.*?\]\(.*?\)"

    WHITESPACE_TAG = r"\s+"

    NUMBERED_LIST_TAG = r"\d+\..+"

    DOT_AT_START_TAG = r"^(\d+)\."


class TextCleaner:
    def __init__(self, config_path="app/products/summarization/files/regex.yaml"):
        self.config = self.load_config(config_path)

    @staticmethod
    def remove_consecutive_dots(text: str) -> str:
        return re.sub(RegexConfig.CONSECUTIVE_DOT_TAG, ".", text)

    @staticmethod
    def add_dot_at_the_end(text: str) -> str:
        return text.rstrip(".") + "."

    @staticmethod
    def clean_html_tags(text: str) -> str:
        return clean_html_tags(text)

    @staticmethod
    def remove_baca_selengkapnya(text: str) -> str:
        return re.sub(RegexConfig.BACA_SELENGKAPNYA_TAG, "", text)

    @staticmethod
    def remove_starting_tag(text: str) -> str:
        return re.sub(RegexConfig.STARTING_TAG, "", text)

    @staticmethod
    def remove_source_viva(text: str) -> str:
        return re.sub(RegexConfig.SOURCE_TAG, "", text, flags=re.MULTILINE)

    @staticmethod
    def remove_gelar(text: str) -> str:
        text = re.sub(RegexConfig.M_ENG_TAG, " M Eng", text)
        return re.sub(RegexConfig.PHD_TAG, " Ph D", text)

    @staticmethod
    def remove_baca_juga(text: str):
        return re.sub(RegexConfig.BACA_JUGA_TAG, "", text, flags=re.MULTILINE)

    @staticmethod
    def remove_consecutive_hashtags(text: str) -> str:
        return re.sub(RegexConfig.CONSECUTIVE_HASHTAG_TAG, "", text)

    @staticmethod
    def remove_links(text: str) -> str:
        return re.sub(RegexConfig.LINKS_TAG, "", text)

    @staticmethod
    def remove_slash_n(text: str) -> str:
        return re.sub(RegexConfig.NEW_LINE_TAG, " ", text)

    @staticmethod
    def remove_tags_with_reuters(text: str) -> str:
        return re.sub(
            RegexConfig.REUTERS_TAG,
            "",
            text,
            flags=re.DOTALL,
        )

    @staticmethod
    def remove_academic_title(text: str) -> str:
        return re.sub(RegexConfig.ONE_TWO_DOT_TAG, r"\1", text)

    @staticmethod
    def remove_markdown_image(text: str) -> str:
        return re.sub(RegexConfig.IMAGE_MARKDOWN_TAG, "", text)

    @staticmethod
    def remove_whitespace(text: str) -> str:
        return re.sub(RegexConfig.WHITESPACE_TAG, " ", text)

    @staticmethod
    def remove_symbols_string(text: str) -> str:
        return "".join(letter for letter in text if letter not in ["*", "\\", "_", "•", ":"])

    @staticmethod
    def fix_numbered_list(text: str) -> str:
        list_items = re.findall(RegexConfig.NUMBERED_LIST_TAG, text)
        adjusted_items = [
            re.sub(RegexConfig.DOT_AT_START_TAG, r"\1", item).rstrip(".") + "."
            for item in list_items
        ]
        for original, adjusted in zip(list_items, adjusted_items):
            text = text.replace(original, adjusted)
        return text

    def apply_function_by_name(self, text, function_name):
        function = getattr(self, function_name, None)
        if function:
            return function(text)
        else:
            raise ValueError(f"Function {function_name} not found")

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def clean_text(self, text: str) -> str:
        """
        Clean the input text using the specified pipeline of functions and return the cleaned text.
        This part is do the thing based on the regex.yaml

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text after applying the pipeline of functions.
        """
        for function_name in self.config.get("pipeline", []):
            text = self.apply_function_by_name(text, function_name)
        return text.strip()


def clean_html_tags(text: str) -> str:
    """Remove Tags of HTML and links, can update this if needed

    Parameters
    ----------
    text : str
        string to clean

    Returns
    -------
    str
        Cleaned string tag of HTML
    """
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    docs_transformed = h.handle(text)
    return remove_symbols_string(docs_transformed)


def remove_symbols_string(text: str) -> str:
    """Clean weird symbol such as * \\ and _. Mainly for parsing pure text

    Parameters
    ----------
    text : str
        string to clean

    Returns
    -------
    str
        Cleaned string from weird symbols
    """
    return "".join(letter for letter in text if letter not in ["*", "\\", "_"])
