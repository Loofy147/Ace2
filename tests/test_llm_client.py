import unittest
from unittest.mock import patch, Mock
import os
from src.ace.core.implementation import Context
from src.ace.llm.client import process_context_with_llm, get_openai_api_key

class TestLLMClient(unittest.TestCase):

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_get_openai_api_key_success(self):
        """Test that the API key is retrieved successfully when set."""
        self.assertEqual(get_openai_api_key(), "test_api_key")

    def test_get_openai_api_key_failure(self):
        """Test that a ValueError is raised when the API key is not set."""
        # Ensure the key is not set for this test
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        with self.assertRaises(ValueError):
            get_openai_api_key()

    @patch('src.ace.llm.client.openai.ChatCompletion.create')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_process_context_with_llm_success(self, mock_create):
        """Test a successful LLM processing call."""
        # Configure the mock to return a predictable response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {'content': '  Mocked LLM summary.  '}
        mock_create.return_value = mock_response

        context = Context(content={"data": "some important content"}, domain="test")
        response = process_context_with_llm(context)

        # Check that the API was called
        mock_create.assert_called_once()
        # Check that the response is cleaned up (stripped of whitespace)
        self.assertEqual(response, "Mocked LLM summary.")

    @patch('src.ace.llm.client.openai.ChatCompletion.create')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_process_context_with_llm_api_error(self, mock_create):
        """Test the handling of an API error."""
        # Configure the mock to raise an exception
        mock_create.side_effect = Exception("API connection failed")

        context = Context(content={"data": "some content"})
        response = process_context_with_llm(context)

        self.assertTrue(response.startswith("Error:"))
        self.assertIn("API connection failed", response)

if __name__ == '__main__':
    unittest.main()
