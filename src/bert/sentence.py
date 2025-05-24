"""
Provides keyword extraction functionality using BERT-based models via KeyBERT.

This module defines the `SentenceBert` class, which encapsulates the KeyBERT
model to extract keywords and keyphrases from a given sentence or text.
It allows customization of extraction parameters like n-gram range, stop words,
and diversity.
"""

from typing import List, Optional, Tuple, Union

from keybert import KeyBERT as KeyBERTModel  # Renamed for clarity to avoid conflict

from src.logger import Logger

logger = Logger()


class SentenceBert:
    """
    A wrapper class for KeyBERT to extract keywords from sentences.

    This class initializes a KeyBERT model and provides a method to extract
    keywords based on sentence similarity and diversification algorithms like MMR.

    Attributes:
        sentence (str): The input sentence from which to extract keywords.
        kw_model (KeyBERTModel): An instance of the KeyBERT model.
    """

    def __init__(self, sentence: str, model_name: Optional[str] = None) -> None:
        """
        Initialize the SentenceBert class with a sentence and optionally a model.

        Args:
            sentence (str): The input sentence or text.
            model_name (Optional[str]): The name or path of the sentence transformer
                                        model to be used by KeyBERT. If None,
                                        KeyBERT's default model (often 'all-MiniLM-L6-v2')
                                        will be used.
        """
        self.sentence: str = sentence
        try:
            if model_name:
                self.kw_model: KeyBERTModel = KeyBERTModel(model=model_name)
            else:
                self.kw_model: KeyBERTModel = KeyBERTModel()
            logger.info(
                f"KeyBERT model initialized (model: {model_name or 'default'})."
            )
        except Exception as e:
            logger.error(f"Failed to initialize KeyBERT model: {e}")
            # Fallback to a default model or raise an error if critical
            # For now, it might still work if KeyBERT() without args has a fallback
            self.kw_model = KeyBERTModel()  # Attempt default initialization
            logger.warning("Attempting KeyBERT default initialization after error.")

    def extract_keywords(
        self,
        top_n: int = 5,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Optional[Union[str, List[str]]] = "english",  # type: ignore
        use_mmr: bool = True,
        diversity: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from the sentence using the KeyBERT model.

        This method utilizes KeyBERT's `extract_keywords` function, which leverages
        sentence embeddings to find the most relevant keywords or keyphrases.
        It can use Maximal Marginal Relevance (MMR) to diversify the results.

        Args:
            top_n (int): The number of top keywords/keyphrases to return.
                         Defaults to 5.
            keyphrase_ngram_range (Tuple[int, int]): The n-gram range for keyphrases.
                                                     For example, (1, 1) means unigrams,
                                                     (1, 2) means unigrams and bigrams.
                                                     Defaults to (1, 1).
            stop_words (Optional[Union[str, List[str]]]): A list of stop words to ignore,
                                                          or a string specifying a language
                                                          (e.g., "english"). Defaults to "english".
                                                          Set to None to disable stop words.
            use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for
                            diversifying results. Defaults to True.
            diversity (float): The diversity factor for MMR, between 0 and 1.
                               Higher values mean more diversity. Defaults to 0.7.

        Returns:
            List[Tuple[str, float]]: A list of tuples, where each tuple contains
                                     a keyword/keyphrase (str) and its relevance
                                     score (float). Returns an empty list if an
                                     error occurs during extraction.
        """
        if not hasattr(self, "kw_model"):
            logger.error("KeyBERT model not initialized. Cannot extract keywords.")
            return []

        try:
            keywords: List[Tuple[str, float]] = self.kw_model.extract_keywords(
                self.sentence,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,  # type: ignore
                top_n=top_n,
                use_mmr=use_mmr,
                diversity=diversity,
            )
            logger.info(f"Successfully extracted {len(keywords)} keywords.")
            return keywords
        except Exception as e:
            logger.error(f"Error during keyword extraction: {e}")
            return []
