import logging
from pathlib import Path
from typing import Optional

from presidio_analyzer.llm_utils import lx

try:
    from langextract import factory as lx_factory
    from langextract.providers import router as lx_router
    from langextract.providers import builtin_registry as lx_builtin
except ImportError:
    lx_factory = None  # type: ignore[misc,assignment]
    lx_router = None  # type: ignore[misc,assignment]
    lx_builtin = None  # type: ignore[misc,assignment]
import langextract_litellm

from presidio_analyzer.predefined_recognizers.third_party.\
    langextract_recognizer import LangExtractRecognizer

logger = logging.getLogger("presidio-analyzer")

# Track if providers have been registered
_providers_registered = False


def _register_langextract_providers():
    """Register LangExtract's built-in providers if not already done."""
    global _providers_registered
    if _providers_registered or lx_router is None or lx_builtin is None:
        return

    for config in lx_builtin.BUILTIN_PROVIDERS:
        lx_router.register_lazy(
            *config["patterns"],
            target=config["target"],
            priority=config.get("priority", 0)
        )
    _providers_registered = True


class ConfigurableLangExtractRecognizer(LangExtractRecognizer):
    """
    Configurable LangExtract recognizer supporting multiple LLM providers.

    This recognizer uses LangExtract's ModelConfig to support any provider:
    - "ollama": Local Ollama server
    - "openai": OpenAI API (or any OpenAI-compatible API via base_url)
    - "gemini": Google Gemini API
    - "litellm": Any LiteLLM-supported model (requires langextract-litellm)

    Example config for LiteLLM (requires: pip install langextract-litellm):
        langextract:
          model:
            # Use litellm model format: provider/model
            model_id: "ollama/qwen2.5:1.5b"
            provider: "litellm"
            provider_kwargs:
              api_base: "http://localhost:11434"  # optional for some providers
    """

    DEFAULT_CONFIG_PATH = (
        Path(__file__).parent.parent.parent
        / "conf"
        / "langextract_config_litellm.yaml"
    )

    # Base providers always available in LangExtract
    BASE_PROVIDERS = {"ollama", "openai", "gemini", "litellm", "LiteLLMLanguageModel"}

    @property
    def supported_providers(self):
        """Return set of supported providers."""
        return self.BASE_PROVIDERS

    def __init__(
        self,
        config_path: Optional[str] = None,
        supported_language: str = "en",
        context: Optional[list] = None
    ):
        """Initialize configurable LangExtract recognizer.

        :param config_path: Path to configuration file (optional).
        :param supported_language: Language this recognizer supports
            (optional, default: "en").
        :param context: List of context words
            (optional, currently not used by LLM recognizers).
        """
        if lx_factory is None:
            raise ImportError(
                "langextract is required for ConfigurableLangExtractRecognizer. "
                "Install it with: pip install langextract"
            )

        # Register LangExtract's built-in providers
        _register_langextract_providers()

        actual_config_path = (
            config_path if config_path else str(self.DEFAULT_CONFIG_PATH)
        )

        super().__init__(
            config_path=actual_config_path,
            name="Configurable LangExtract PII",
            supported_language=supported_language
        )

        model_config = self.config.get("model", {})
        self.provider = model_config.get("provider")
        self.provider_kwargs = model_config.get("provider_kwargs", {})

        if not self.provider:
            raise ValueError(
                "Model configuration must contain 'provider'. "
                f"Supported: {self.supported_providers}"
            )

        if self.provider not in self.supported_providers:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                f"Supported: {self.supported_providers}"
            )


        self.lx_config = lx_factory.ModelConfig(
            model_id=self.model_id,
            provider=self.provider,
            provider_kwargs=self.provider_kwargs
        )

    def _call_langextract(self, **kwargs):
        """Call LLM through LangExtract with configured provider."""
        try:
            extract_params = {
                "text_or_documents": kwargs.pop("text"),
                "prompt_description": kwargs.pop("prompt"),
                "examples": kwargs.pop("examples"),
                "config": self.lx_config,
            }

            extract_params.update(kwargs)

            return lx.extract(**extract_params)
        except Exception:
            logger.exception(
                "LangExtract extraction failed (provider='%s', model='%s')",
                self.provider, self.model_id
            )
            raise


# Alias for backward compatibility
LitellmLangExtractRecognizer = ConfigurableLangExtractRecognizer
