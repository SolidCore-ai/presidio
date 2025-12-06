"""Tests for ConfigurableLangExtractRecognizer."""
import pytest
from unittest.mock import Mock, patch, MagicMock


def create_test_config(
    supported_entities=None,
    entity_mappings=None,
    model_id="litellm/ollama/qwen2.5:1.5b",
    provider="LiteLLMLanguageModel",
    provider_kwargs=None,
    temperature=0.0,
    min_score=0.5,
    labels_to_ignore=None,
    enable_generic_consolidation=True
):
    """Create test config for ConfigurableLangExtractRecognizer."""
    if supported_entities is None:
        supported_entities = ["PERSON", "EMAIL_ADDRESS"]
    if entity_mappings is None:
        entity_mappings = {"person": "PERSON", "email": "EMAIL_ADDRESS"}
    if labels_to_ignore is None:
        labels_to_ignore = []
    if provider_kwargs is None:
        provider_kwargs = {}

    return {
        "lm_recognizer": {
            "supported_entities": supported_entities,
            "labels_to_ignore": labels_to_ignore,
            "enable_generic_consolidation": enable_generic_consolidation,
            "min_score": min_score,
        },
        "langextract": {
            "prompt_file": "presidio-analyzer/presidio_analyzer/conf/langextract_prompts/default_pii_phi_prompt.j2",
            "examples_file": "presidio-analyzer/presidio_analyzer/conf/langextract_prompts/default_pii_phi_examples.yaml",
            "model": {
                "model_id": model_id,
                "provider": provider,
                "provider_kwargs": provider_kwargs,
                "temperature": temperature,
            },
            "entity_mappings": entity_mappings,
        }
    }


@pytest.fixture
def mock_langextract():
    """Mock langextract and its dependencies."""
    with patch('presidio_analyzer.llm_utils.langextract_helper.lx', Mock()):
        with patch(
            'presidio_analyzer.predefined_recognizers.third_party.'
            'litellm_langextract_recognizer.lx_factory'
        ) as mock_factory:
            with patch(
                'presidio_analyzer.predefined_recognizers.third_party.'
                'litellm_langextract_recognizer.lx_router'
            ) as mock_router:
                with patch(
                    'presidio_analyzer.predefined_recognizers.third_party.'
                    'litellm_langextract_recognizer.lx_builtin'
                ) as mock_builtin:
                    # Setup mock factory
                    mock_factory.ModelConfig = Mock(return_value=Mock())
                    mock_builtin.BUILTIN_PROVIDERS = []
                    yield {
                        'factory': mock_factory,
                        'router': mock_router,
                        'builtin': mock_builtin
                    }


class TestConfigurableLangExtractRecognizerCallLangextract:
    """Test the _call_langextract method."""

    @pytest.fixture
    def mock_recognizer(self, tmp_path, mock_langextract):
        """Fixture to create a mocked recognizer."""
        import yaml

        config = create_test_config(
            model_id="test-model",
            provider="openai",
            provider_kwargs={"api_key": "test"}
        )

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        from presidio_analyzer.predefined_recognizers.third_party.\
            litellm_langextract_recognizer import ConfigurableLangExtractRecognizer

        return ConfigurableLangExtractRecognizer(config_path=str(config_file))

    def test_call_langextract_passes_config_to_extract(self, mock_recognizer):
        """Test that _call_langextract passes the ModelConfig to lx.extract."""
        mock_result = Mock()
        mock_result.extractions = []

        with patch(
            'presidio_analyzer.predefined_recognizers.third_party.'
            'litellm_langextract_recognizer.lx.extract',
            return_value=mock_result
        ) as mock_extract:
            mock_recognizer._call_langextract(
                text="test text",
                prompt="test prompt",
                examples=[]
            )

            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args[1]
            assert "config" in call_kwargs
            assert call_kwargs["text_or_documents"] == "test text"
            assert call_kwargs["prompt_description"] == "test prompt"
            assert call_kwargs["examples"] == []

    def test_call_langextract_logs_error_on_failure(self, mock_recognizer, caplog):
        """Test that errors are logged when extraction fails."""
        import logging

        with patch(
            'presidio_analyzer.predefined_recognizers.third_party.'
            'litellm_langextract_recognizer.lx.extract',
            side_effect=Exception("API Error")
        ):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(Exception, match="API Error"):
                    mock_recognizer._call_langextract(
                        text="test",
                        prompt="test",
                        examples=[]
                    )

            assert "LangExtract extraction failed" in caplog.text


class TestAnalyzeWithMockedModelResponse:
    """
    Integration-style tests that simulate real LLM responses.
    
    These tests mock the LangExtract extraction response to simulate
    what a real LLM would return, testing the full analyze pipeline
    including entity mapping and score assignment.
    """

    @pytest.fixture
    def recognizer_with_mock_extract(self, tmp_path, mock_langextract):
        """Create recognizer with mocked lx.extract for simulating LLM responses."""
        import yaml

        config = create_test_config(
            supported_entities=[
                "PERSON", "EMAIL_ADDRESS", "LOCATION",
                "PHONE_NUMBER", "ORGANIZATION"
            ],
            entity_mappings={
                "person": "PERSON",
                "name": "PERSON",
                "email": "EMAIL_ADDRESS",
                "location": "LOCATION",
                "address": "LOCATION",
                "phone": "PHONE_NUMBER",
                "organization": "ORGANIZATION",
                "company": "ORGANIZATION"
            },
            model_id="gpt-4o-mini",
            provider="openai",
            provider_kwargs={"api_key": "test-key", "base_url": "http://test"},
            min_score=0.5,
            enable_generic_consolidation=True
        )

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        from presidio_analyzer.predefined_recognizers.third_party.\
            litellm_langextract_recognizer import ConfigurableLangExtractRecognizer

        return ConfigurableLangExtractRecognizer(config_path=str(config_file))

    def _create_mock_extraction(
        self, extraction_class, extraction_text, start_pos, end_pos,
        alignment_status="MATCH_EXACT", attributes=None
    ):
        """Helper to create a mock LangExtract extraction object."""
        mock_extraction = Mock()
        mock_extraction.extraction_class = extraction_class
        mock_extraction.extraction_text = extraction_text
        mock_extraction.char_interval = Mock()
        mock_extraction.char_interval.start_pos = start_pos
        mock_extraction.char_interval.end_pos = end_pos
        mock_extraction.alignment_status = alignment_status
        mock_extraction.attributes = attributes or {}
        return mock_extraction

    def test_analyze_detects_person_and_email(self, recognizer_with_mock_extract):
        """
        Test: Detect PERSON and EMAIL_ADDRESS entities.
        
        Simulates LLM response for: "Contact John Smith at john.smith@example.com"
        Expected: PERSON at 8-18, EMAIL_ADDRESS at 22-44
        """
        text = "Contact John Smith at john.smith@example.com"

        # Simulate LLM extraction response
        mock_result = Mock()
        mock_result.extractions = [
            self._create_mock_extraction(
                extraction_class="person",
                extraction_text="John Smith",
                start_pos=8,
                end_pos=18,
                alignment_status="MATCH_EXACT"
            ),
            self._create_mock_extraction(
                extraction_class="email",
                extraction_text="john.smith@example.com",
                start_pos=22,
                end_pos=44,
                alignment_status="MATCH_EXACT"
            )
        ]

        with patch(
            'presidio_analyzer.predefined_recognizers.third_party.'
            'litellm_langextract_recognizer.lx.extract',
            return_value=mock_result
        ):
            results = recognizer_with_mock_extract.analyze(
                text,
                entities=["PERSON", "EMAIL_ADDRESS"]
            )

        assert len(results) == 2

        # Verify PERSON detection
        person_results = [r for r in results if r.entity_type == "PERSON"]
        assert len(person_results) == 1
        assert person_results[0].start == 8
        assert person_results[0].end == 18
        assert text[8:18] == "John Smith"
        assert person_results[0].score == 0.95  # MATCH_EXACT score

        # Verify EMAIL_ADDRESS detection
        email_results = [r for r in results if r.entity_type == "EMAIL_ADDRESS"]
        assert len(email_results) == 1
        assert email_results[0].start == 22
        assert email_results[0].end == 44
        assert text[22:44] == "john.smith@example.com"

    def test_analyze_detects_multiple_entity_types(self, recognizer_with_mock_extract):
        """
        Test: Detect multiple entity types in a complex sentence.
        
        Simulates LLM response for:
        "Jane Doe works at Acme Corp in San Francisco. Call 555-123-4567."
        """
        text = "Jane Doe works at Acme Corp in San Francisco. Call 555-123-4567."

        mock_result = Mock()
        mock_result.extractions = [
            self._create_mock_extraction(
                extraction_class="person",
                extraction_text="Jane Doe",
                start_pos=0,
                end_pos=8,
                alignment_status="MATCH_EXACT"
            ),
            self._create_mock_extraction(
                extraction_class="company",  # Uses entity mapping
                extraction_text="Acme Corp",
                start_pos=18,
                end_pos=27,
                alignment_status="MATCH_EXACT"
            ),
            self._create_mock_extraction(
                extraction_class="location",
                extraction_text="San Francisco",
                start_pos=31,
                end_pos=44,
                alignment_status="MATCH_EXACT"
            ),
            self._create_mock_extraction(
                extraction_class="phone",
                extraction_text="555-123-4567",
                start_pos=51,
                end_pos=63,
                alignment_status="MATCH_FUZZY"  # Lower score
            )
        ]

        with patch(
            'presidio_analyzer.predefined_recognizers.third_party.'
            'litellm_langextract_recognizer.lx.extract',
            return_value=mock_result
        ):
            results = recognizer_with_mock_extract.analyze(text)

        assert len(results) == 4

        # Check entity types are correctly mapped
        entity_types = {r.entity_type for r in results}
        assert entity_types == {"PERSON", "ORGANIZATION", "LOCATION", "PHONE_NUMBER"}

        # Verify company -> ORGANIZATION mapping
        org_results = [r for r in results if r.entity_type == "ORGANIZATION"]
        assert len(org_results) == 1
        assert text[org_results[0].start:org_results[0].end] == "Acme Corp"

        # Verify MATCH_FUZZY has lower score
        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        assert phone_results[0].score == 0.80  # MATCH_FUZZY score

    def test_analyze_handles_unknown_entity_with_generic_consolidation(
        self, recognizer_with_mock_extract
    ):
        """
        Test: Unknown entity types are consolidated to GENERIC_PII_ENTITY.
        
        When LLM returns an entity class not in entity_mappings,
        it should be converted to GENERIC_PII_ENTITY.
        """
        text = "Patient ID: ABC-12345-XYZ"

        mock_result = Mock()
        mock_result.extractions = [
            self._create_mock_extraction(
                extraction_class="patient_id",  # Not in entity_mappings
                extraction_text="ABC-12345-XYZ",
                start_pos=12,
                end_pos=25,
                alignment_status="MATCH_EXACT"
            )
        ]

        with patch(
            'presidio_analyzer.predefined_recognizers.third_party.'
            'litellm_langextract_recognizer.lx.extract',
            return_value=mock_result
        ):
            # Request all entities including GENERIC_PII_ENTITY
            results = recognizer_with_mock_extract.analyze(text)

        assert len(results) == 1
        assert results[0].entity_type == "GENERIC_PII_ENTITY"
        assert results[0].start == 12
        assert results[0].end == 25
        assert text[12:25] == "ABC-12345-XYZ"

        # Verify original type is preserved in metadata
        assert results[0].recognition_metadata["original_entity_type"] == "PATIENT_ID"
