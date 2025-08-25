"""
Multilingual Documentation System Module
Handles internationalization and localization for documentation
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
import re
from datetime import datetime


class SupportedLanguage(Enum):
    """Supported languages for documentation"""
    ENGLISH = "en"
    PORTUGUESE_BR = "pt-BR" 
    KOREAN = "ko"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    RUSSIAN = "ru"


@dataclass
class Translation:
    """Represents a translation for a specific term or phrase"""
    key: str
    source_text: str
    translated_text: str
    language: SupportedLanguage
    context: str = ""
    reviewed: bool = False
    translator: str = ""
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DocumentTranslation:
    """Represents a translated document"""
    original_path: str
    translated_path: str
    language: SupportedLanguage
    translation_status: str = "pending"  # pending, in_progress, completed, reviewed
    completion_percentage: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    translator_notes: List[str] = field(default_factory=list)


@dataclass
class LanguageConfig:
    """Configuration for a specific language"""
    language: SupportedLanguage
    display_name: str
    rtl: bool = False  # Right-to-left languages
    date_format: str = "%Y-%m-%d"
    number_format: str = "1,234.56"
    currency_format: str = "${amount}"
    locale_code: str = ""
    fallback_language: Optional[SupportedLanguage] = None


class MultilingualDocsSystem:
    """Manages multilingual documentation system"""
    
    def __init__(self):
        self.languages = self._initialize_language_configs()
        self.translations = {}  # key -> {lang -> translation}
        self.document_translations = {}
        self.navigation_translations = {}
        self.ui_translations = {}
        
    def _initialize_language_configs(self) -> Dict[SupportedLanguage, LanguageConfig]:
        """Initialize language configurations"""
        return {
            SupportedLanguage.ENGLISH: LanguageConfig(
                language=SupportedLanguage.ENGLISH,
                display_name="English",
                locale_code="en-US",
                date_format="%m/%d/%Y",
                currency_format="${amount}"
            ),
            SupportedLanguage.PORTUGUESE_BR: LanguageConfig(
                language=SupportedLanguage.PORTUGUESE_BR,
                display_name="Português (Brasil)",
                locale_code="pt-BR",
                date_format="%d/%m/%Y",
                currency_format="R$ {amount}",
                fallback_language=SupportedLanguage.ENGLISH
            ),
            SupportedLanguage.KOREAN: LanguageConfig(
                language=SupportedLanguage.KOREAN,
                display_name="한국어",
                locale_code="ko-KR",
                date_format="%Y/%m/%d",
                currency_format="₩{amount}",
                fallback_language=SupportedLanguage.ENGLISH
            ),
            SupportedLanguage.SPANISH: LanguageConfig(
                language=SupportedLanguage.SPANISH,
                display_name="Español",
                locale_code="es-ES",
                date_format="%d/%m/%Y",
                currency_format="€{amount}",
                fallback_language=SupportedLanguage.ENGLISH
            ),
            SupportedLanguage.FRENCH: LanguageConfig(
                language=SupportedLanguage.FRENCH,
                display_name="Français",
                locale_code="fr-FR",
                date_format="%d/%m/%Y",
                currency_format="{amount} €",
                fallback_language=SupportedLanguage.ENGLISH
            ),
            SupportedLanguage.JAPANESE: LanguageConfig(
                language=SupportedLanguage.JAPANESE,
                display_name="日本語",
                locale_code="ja-JP",
                date_format="%Y年%m月%d日",
                currency_format="¥{amount}",
                fallback_language=SupportedLanguage.ENGLISH
            ),
            SupportedLanguage.CHINESE_SIMPLIFIED: LanguageConfig(
                language=SupportedLanguage.CHINESE_SIMPLIFIED,
                display_name="简体中文",
                locale_code="zh-CN",
                date_format="%Y年%m月%d日",
                currency_format="¥{amount}",
                fallback_language=SupportedLanguage.ENGLISH
            )
        }
    
    def add_translation(self, key: str, source_text: str, 
                       language: SupportedLanguage, translated_text: str,
                       context: str = "") -> Translation:
        """Add a translation for a specific key"""
        translation = Translation(
            key=key,
            source_text=source_text,
            translated_text=translated_text,
            language=language,
            context=context
        )
        
        if key not in self.translations:
            self.translations[key] = {}
        
        self.translations[key][language] = translation
        return translation
    
    def get_translation(self, key: str, language: SupportedLanguage) -> Optional[str]:
        """Get translation for a key in specific language"""
        if key in self.translations and language in self.translations[key]:
            return self.translations[key][language].translated_text
        
        # Try fallback language
        if language in self.languages:
            fallback_lang = self.languages[language].fallback_language
            if fallback_lang and key in self.translations and fallback_lang in self.translations[key]:
                return self.translations[key][fallback_lang].translated_text
        
        # Return key as fallback
        return key
    
    def extract_translatable_strings(self, content: str, content_type: str = "markdown") -> List[str]:
        """Extract translatable strings from content"""
        translatable_strings = []
        
        if content_type.lower() == "markdown":
            # Extract headers
            headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            translatable_strings.extend(headers)
            
            # Extract regular text (excluding code blocks)
            # Remove code blocks first
            content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            content_no_inline_code = re.sub(r'`[^`]+`', '', content_no_code)
            
            # Extract paragraphs
            paragraphs = re.findall(r'^[^#\-\*\|\s][^\n]*$', content_no_inline_code, re.MULTILINE)
            for paragraph in paragraphs:
                if len(paragraph.strip()) > 10:  # Ignore very short strings
                    translatable_strings.append(paragraph.strip())
            
            # Extract list items
            list_items = re.findall(r'^\s*[\-\*]\s+(.+)$', content, re.MULTILINE)
            translatable_strings.extend(list_items)
            
        elif content_type.lower() == "json":
            # Extract from JSON structure (for navigation, UI elements)
            try:
                data = json.loads(content)
                translatable_strings.extend(self._extract_from_json(data))
            except json.JSONDecodeError:
                pass
        
        # Clean and deduplicate
        cleaned_strings = []
        for string in translatable_strings:
            cleaned = string.strip()
            if cleaned and len(cleaned) > 2 and cleaned not in cleaned_strings:
                cleaned_strings.append(cleaned)
        
        return cleaned_strings
    
    def _extract_from_json(self, data: Any) -> List[str]:
        """Recursively extract translatable strings from JSON data"""
        strings = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Keys that typically contain translatable content
                if key in ['title', 'description', 'label', 'name', 'tab', 'group', 'anchor']:
                    if isinstance(value, str) and len(value) > 2:
                        strings.append(value)
                else:
                    strings.extend(self._extract_from_json(value))
        elif isinstance(data, list):
            for item in data:
                strings.extend(self._extract_from_json(item))
        
        return strings
    
    def create_translation_template(self, source_language: SupportedLanguage,
                                  target_language: SupportedLanguage,
                                  content: str) -> Dict[str, Any]:
        """Create translation template for translators"""
        translatable_strings = self.extract_translatable_strings(content)
        
        template = {
            "source_language": source_language.value,
            "target_language": target_language.value,
            "created_date": datetime.now().isoformat(),
            "total_strings": len(translatable_strings),
            "strings_to_translate": []
        }
        
        for i, string in enumerate(translatable_strings):
            template["strings_to_translate"].append({
                "id": f"string_{i+1}",
                "source_text": string,
                "translated_text": "",
                "context": "",
                "notes": "",
                "status": "pending"
            })
        
        return template
    
    def generate_multilingual_navigation(self, base_navigation: Dict[str, Any],
                                       language: SupportedLanguage) -> Dict[str, Any]:
        """Generate navigation structure for specific language"""
        translated_nav = json.loads(json.dumps(base_navigation))  # Deep copy
        
        def translate_navigation_item(item: Any) -> Any:
            if isinstance(item, dict):
                # Translate common navigation keys
                for key in ['tab', 'group', 'anchor', 'title', 'description']:
                    if key in item and isinstance(item[key], str):
                        translation_key = f"nav.{key}.{item[key].lower().replace(' ', '_')}"
                        item[key] = self.get_translation(translation_key, language)
                
                # Recursively process nested items
                for key, value in item.items():
                    item[key] = translate_navigation_item(value)
                
            elif isinstance(item, list):
                return [translate_navigation_item(sub_item) for sub_item in item]
            
            return item
        
        return translate_navigation_item(translated_nav)
    
    def create_language_switcher_config(self) -> Dict[str, Any]:
        """Create configuration for language switcher"""
        return {
            "languages": [
                {
                    "code": lang_config.language.value,
                    "name": lang_config.display_name,
                    "locale": lang_config.locale_code,
                    "rtl": lang_config.rtl
                }
                for lang_config in self.languages.values()
            ],
            "default_language": SupportedLanguage.ENGLISH.value,
            "fallback_language": SupportedLanguage.ENGLISH.value,
            "detection_order": ["path", "cookie", "header", "browser"]
        }
    
    def validate_translations(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Validate translations for completeness and quality"""
        validation_report = {
            "language": language.value,
            "total_keys": len(self.translations),
            "translated_keys": 0,
            "missing_translations": [],
            "empty_translations": [],
            "potential_issues": [],
            "completion_percentage": 0.0
        }
        
        for key, translations in self.translations.items():
            if language in translations:
                translation = translations[language]
                validation_report["translated_keys"] += 1
                
                # Check for empty translations
                if not translation.translated_text.strip():
                    validation_report["empty_translations"].append(key)
                
                # Check for potential issues
                source_len = len(translation.source_text)
                trans_len = len(translation.translated_text)
                
                # Flag if translated text is significantly different in length
                if trans_len < source_len * 0.3 or trans_len > source_len * 3:
                    validation_report["potential_issues"].append({
                        "key": key,
                        "issue": "Length discrepancy",
                        "source_length": source_len,
                        "translation_length": trans_len
                    })
                
                # Check for untranslated technical terms
                if translation.source_text == translation.translated_text:
                    validation_report["potential_issues"].append({
                        "key": key,
                        "issue": "Identical to source",
                        "text": translation.source_text
                    })
            else:
                validation_report["missing_translations"].append(key)
        
        # Calculate completion percentage
        if validation_report["total_keys"] > 0:
            validation_report["completion_percentage"] = (
                validation_report["translated_keys"] / validation_report["total_keys"] * 100
            )
        
        return validation_report
    
    def export_translations_for_locale(self, language: SupportedLanguage,
                                     format: str = "json") -> str:
        """Export translations in format suitable for i18n libraries"""
        translation_data = {}
        
        for key, translations in self.translations.items():
            if language in translations:
                # Create nested structure from dot notation keys
                keys = key.split('.')
                current_level = translation_data
                
                for i, key_part in enumerate(keys):
                    if i == len(keys) - 1:
                        # Last key, set the value
                        current_level[key_part] = translations[language].translated_text
                    else:
                        # Intermediate key, create nested dict
                        if key_part not in current_level:
                            current_level[key_part] = {}
                        current_level = current_level[key_part]
        
        if format.lower() == "json":
            return json.dumps(translation_data, indent=2, ensure_ascii=False)
        elif format.lower() in ["yaml", "yml"]:
            return yaml.dump(translation_data, allow_unicode=True, default_flow_style=False)
        
        return str(translation_data)
    
    def create_localized_url_structure(self, base_structure: Dict[str, str],
                                     language: SupportedLanguage) -> Dict[str, str]:
        """Create localized URL structure"""
        localized_urls = {}
        lang_code = language.value
        
        for path, title in base_structure.items():
            # Translate path segments if needed
            if language != SupportedLanguage.ENGLISH:
                # For non-English languages, prefix with language code
                localized_path = f"/{lang_code}{path}"
            else:
                localized_path = path
            
            # Translate the title
            translation_key = f"url.{path.replace('/', '.').strip('.')}"
            localized_title = self.get_translation(translation_key, language)
            
            localized_urls[localized_path] = localized_title
        
        return localized_urls
    
    def generate_language_metadata(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Generate metadata for a specific language"""
        if language not in self.languages:
            return {}
        
        config = self.languages[language]
        return {
            "language": language.value,
            "display_name": config.display_name,
            "locale": config.locale_code,
            "direction": "rtl" if config.rtl else "ltr",
            "date_format": config.date_format,
            "number_format": config.number_format,
            "currency_format": config.currency_format,
            "fallback_language": config.fallback_language.value if config.fallback_language else None
        }
    
    def create_translation_memory(self) -> Dict[str, Any]:
        """Create translation memory for reuse and consistency"""
        memory = {
            "created_date": datetime.now().isoformat(),
            "total_translations": sum(len(translations) for translations in self.translations.values()),
            "languages": [lang.value for lang in self.languages.keys()],
            "translation_pairs": []
        }
        
        for key, translations in self.translations.items():
            if SupportedLanguage.ENGLISH in translations:
                source_text = translations[SupportedLanguage.ENGLISH].translated_text
                
                for language, translation in translations.items():
                    if language != SupportedLanguage.ENGLISH:
                        memory["translation_pairs"].append({
                            "source_language": SupportedLanguage.ENGLISH.value,
                            "target_language": language.value,
                            "source_text": source_text,
                            "target_text": translation.translated_text,
                            "context": translation.context,
                            "key": key
                        })
        
        return memory
    
    def suggest_similar_translations(self, source_text: str, 
                                   target_language: SupportedLanguage,
                                   similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Suggest similar translations from translation memory"""
        suggestions = []
        
        # Simple similarity based on word overlap (can be improved with NLP)
        source_words = set(source_text.lower().split())
        
        for translations in self.translations.values():
            if target_language in translations:
                translation = translations[target_language]
                candidate_words = set(translation.source_text.lower().split())
                
                if len(candidate_words) > 0:
                    similarity = len(source_words & candidate_words) / len(source_words | candidate_words)
                    
                    if similarity >= similarity_threshold:
                        suggestions.append({
                            "source_text": translation.source_text,
                            "translated_text": translation.translated_text,
                            "similarity_score": similarity,
                            "context": translation.context
                        })
        
        # Sort by similarity score
        suggestions.sort(key=lambda x: x["similarity_score"], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions