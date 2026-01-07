"""Language support and character set definitions for OCR training.

This module provides language-specific character sets for font compatibility checking
and supports multiple languages via ISO 639-1/639-3 codes.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LanguageCharacterSet:
    """Character set definition for a language.

    Attributes:
        iso_639_1: Two-letter ISO 639-1 code (e.g., "is" for Icelandic)
        iso_639_3: Three-letter ISO 639-3 code (e.g., "isl" for Icelandic)
        name_english: English name of the language
        name_native: Native name of the language
        special_characters: String of special characters to check for font compatibility
        description: Optional description of the character set
    """

    iso_639_1: str
    iso_639_3: str
    name_english: str
    name_native: str
    special_characters: str
    description: str = ""

    def __post_init__(self):
        """Validate ISO codes on initialization."""
        if len(self.iso_639_1) != 2:
            raise ValueError(
                f"ISO 639-1 code must be 2 characters, got: {self.iso_639_1}"
            )
        if len(self.iso_639_3) != 3:
            raise ValueError(
                f"ISO 639-3 code must be 3 characters, got: {self.iso_639_3}"
            )
        if not self.special_characters:
            raise ValueError("special_characters cannot be empty")


class LanguageRegistry:
    """Registry of supported languages and their character sets.

    This class maintains a registry of languages with their special character sets
    for font compatibility checking. Languages can be accessed by ISO 639-1 or
    ISO 639-3 codes.

    Example:
        >>> lang = LanguageRegistry.get_language("is")
        >>> print(lang.special_characters)
        ÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö

        >>> # Add custom language
        >>> LanguageRegistry.add_custom_language(LanguageCharacterSet(
        ...     iso_639_1="ja",
        ...     iso_639_3="jpn",
        ...     name_english="Japanese",
        ...     name_native="日本語",
        ...     special_characters="あいうえお漢字"
        ... ))
    """

    LANGUAGES: ClassVar[dict[str, LanguageCharacterSet]] = {
        "is": LanguageCharacterSet(
            iso_639_1="is",
            iso_639_3="isl",
            name_english="Icelandic",
            name_native="Íslenska",
            special_characters="ÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö",
            description="Icelandic special characters including eth (Ðð), thorn (Þþ), and accented vowels",
        ),
        "de": LanguageCharacterSet(
            iso_639_1="de",
            iso_639_3="deu",
            name_english="German",
            name_native="Deutsch",
            special_characters="ÄäÖöÜüß",
            description="German umlauts (Ä, Ö, Ü) and eszett (ß)",
        ),
        "fr": LanguageCharacterSet(
            iso_639_1="fr",
            iso_639_3="fra",
            name_english="French",
            name_native="Français",
            special_characters="ÀàÂâÆæÇçÉéÈèÊêËëÎîÏïÔôŒœÙùÛûÜüŸÿ",
            description="French accented characters, cedilla, and ligatures (Æ, Œ)",
        ),
        "es": LanguageCharacterSet(
            iso_639_1="es",
            iso_639_3="spa",
            name_english="Spanish",
            name_native="Español",
            special_characters="ÁáÉéÍíÑñÓóÚúÜü¿¡",
            description="Spanish accented vowels, eñe (Ñ), and inverted punctuation (¿¡)",
        ),
        "pl": LanguageCharacterSet(
            iso_639_1="pl",
            iso_639_3="pol",
            name_english="Polish",
            name_native="Polski",
            special_characters="ĄąĆćĘęŁłŃńÓóŚśŹźŻż",
            description="Polish letters with diacritical marks (ogonek, acute, kreska)",
        ),
        "cs": LanguageCharacterSet(
            iso_639_1="cs",
            iso_639_3="ces",
            name_english="Czech",
            name_native="Čeština",
            special_characters="ÁáČčĎďÉéĚěÍíŇňÓóŘřŠšŤťÚúŮůÝýŽž",
            description="Czech letters with acute accent, caron (háček), and ring",
        ),
        "pt": LanguageCharacterSet(
            iso_639_1="pt",
            iso_639_3="por",
            name_english="Portuguese",
            name_native="Português",
            special_characters="ÁáÂâÃãÀàÇçÉéÊêÍíÓóÔôÕõÚú",
            description="Portuguese accented vowels, cedilla, and tilde",
        ),
        "sv": LanguageCharacterSet(
            iso_639_1="sv",
            iso_639_3="swe",
            name_english="Swedish",
            name_native="Svenska",
            special_characters="ÅåÄäÖö",
            description="Swedish letters Å, Ä, and Ö",
        ),
        "no": LanguageCharacterSet(
            iso_639_1="no",
            iso_639_3="nor",
            name_english="Norwegian",
            name_native="Norsk",
            special_characters="ÆæØøÅå",
            description="Norwegian letters Æ, Ø, and Å",
        ),
        "da": LanguageCharacterSet(
            iso_639_1="da",
            iso_639_3="dan",
            name_english="Danish",
            name_native="Dansk",
            special_characters="ÆæØøÅå",
            description="Danish letters Æ, Ø, and Å",
        ),
    }

    @classmethod
    def get_language(cls, code: str) -> LanguageCharacterSet:
        """Get language character set by ISO 639-1 or ISO 639-3 code.

        Args:
            code: ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code

        Returns:
            LanguageCharacterSet for the requested language

        Raises:
            NotImplementedError: If the language code is not supported

        Example:
            >>> lang = LanguageRegistry.get_language("is")
            >>> print(lang.name_english)
            Icelandic
        """
        # Try ISO 639-1 first (most common)
        if code in cls.LANGUAGES:
            return cls.LANGUAGES[code]

        # Try ISO 639-3
        for lang in cls.LANGUAGES.values():
            if lang.iso_639_3 == code:
                return lang

        # Not found - raise NotImplementedError as requested
        raise NotImplementedError(
            f"Language '{code}' is not supported. "
            f"Supported languages: {', '.join(cls.list_supported_languages())}. "
            f"To add a custom language, use LanguageRegistry.add_custom_language()."
        )

    @classmethod
    def list_supported_languages(cls) -> list[str]:
        """Return list of supported ISO 639-1 codes.

        Returns:
            List of two-letter ISO 639-1 language codes

        Example:
            >>> codes = LanguageRegistry.list_supported_languages()
            >>> "is" in codes
            True
        """
        return sorted(cls.LANGUAGES.keys())

    @classmethod
    def get_language_info(cls) -> list[dict[str, str]]:
        """Get detailed information about all supported languages.

        Returns:
            List of dictionaries with language information

        Example:
            >>> info = LanguageRegistry.get_language_info()
            >>> info[0]['iso_639_1']
            'cs'
        """
        return [
            {
                "iso_639_1": lang.iso_639_1,
                "iso_639_3": lang.iso_639_3,
                "name_english": lang.name_english,
                "name_native": lang.name_native,
                "description": lang.description,
            }
            for lang in sorted(cls.LANGUAGES.values(), key=lambda x: x.iso_639_1)
        ]

    @classmethod
    def add_custom_language(cls, lang: LanguageCharacterSet) -> None:
        """Add a custom language to the registry at runtime.

        This allows users to extend language support without modifying the source code.

        Args:
            lang: LanguageCharacterSet instance defining the new language

        Raises:
            ValueError: If ISO codes are invalid or language already exists

        Example:
            >>> LanguageRegistry.add_custom_language(LanguageCharacterSet(
            ...     iso_639_1="ja",
            ...     iso_639_3="jpn",
            ...     name_english="Japanese",
            ...     name_native="日本語",
            ...     special_characters="あいうえお漢字平仮名片仮名"
            ... ))
        """
        if lang.iso_639_1 in cls.LANGUAGES:
            raise ValueError(
                f"Language '{lang.iso_639_1}' already exists in registry. "
                f"Use a different ISO code or remove the existing entry first."
            )

        # Check for ISO 639-3 conflicts
        for existing_lang in cls.LANGUAGES.values():
            if existing_lang.iso_639_3 == lang.iso_639_3:
                raise ValueError(
                    f"ISO 639-3 code '{lang.iso_639_3}' already exists for "
                    f"language '{existing_lang.iso_639_1}'"
                )

        cls.LANGUAGES[lang.iso_639_1] = lang

    @classmethod
    def remove_language(cls, code: str) -> None:
        """Remove a language from the registry.

        Args:
            code: ISO 639-1 code of the language to remove

        Raises:
            KeyError: If language code doesn't exist

        Example:
            >>> LanguageRegistry.remove_language("ja")
        """
        if code not in cls.LANGUAGES:
            raise KeyError(f"Language '{code}' not found in registry")
        del cls.LANGUAGES[code]
