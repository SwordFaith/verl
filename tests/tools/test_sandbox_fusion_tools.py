# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, Mock, patch

import pytest

from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.schemas import OpenAIFunctionToolSchema


class TestSandboxFusionToolRunCode:
    """Test class for SandboxFusionTool run_code mode error handling optimization."""

    @pytest.fixture
    def tool_config(self):
        """Default tool configuration."""
        return {"num_workers": 1, "rate_limit": 1, "default_timeout": 30, "cell_timeout": 10, "default_language": "python", "enable_global_rate_limit": False, "sandbox_fusion_url": "http://mock-sandbox:8000", "mode": "run_code"}

    @pytest.fixture
    def tool_schema(self):
        """Default tool schema."""
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for execute code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "code needs to be execute",
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        )

    @pytest.fixture
    def mock_tool(self, tool_config, tool_schema):
        """Create a mocked SandboxFusionTool."""
        with patch("verl.tools.sandbox_fusion_tools.init_execution_pool"):
            tool = SandboxFusionTool(tool_config, tool_schema)
            tool.execution_pool = Mock()
            return tool

    def test_successful_execution(self, mock_tool):
        """Test successful code execution returns stdout."""
        # Mock successful metadata
        mock_metadata = {"api_request_error": None, "api_status": "Success", "run_status": "Finished", "exit_code": 0, "stdout": "Hello, World!\n", "stderr": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(True, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print('Hello, World!')", timeout=30)
            assert result == "Hello, World!\n"
            assert success is True

    def test_api_request_error(self, mock_tool):
        """Test API request error handling."""
        mock_metadata = {"api_request_error": "API Call Failed: Connection timeout", "api_status": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-1, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert "Error in calling code interpreter: API Call Failed: Connection timeout" in result
            assert success is False

    def test_gateway_timeout_retry(self, mock_tool):
        """Test retry mechanism for Gateway Timeout errors."""
        # First call fails with gateway timeout, second succeeds
        mock_metadata_fail = {"api_request_error": "Gateway Timeout (504)", "api_status": None}
        mock_metadata_success = {"api_request_error": None, "api_status": "Success", "run_status": "Finished", "exit_code": 0, "stdout": "Success after retry\n", "stderr": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", side_effect=[(-1, mock_metadata_fail), (True, mock_metadata_success)]), patch("time.sleep"), patch("random.uniform", return_value=1.0):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert result == "Success after retry\n"
            assert success is True

    def test_sandbox_error(self, mock_tool):
        """Test sandbox error handling."""
        mock_metadata = {"api_request_error": None, "api_status": "SandboxError", "api_response": "Internal sandbox error occurred"}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-1, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert "Error in calling code interpreter: Sandbox error occurred: Internal sandbox error occurred" in result
            assert success is False

    def test_compile_error(self, mock_tool):
        """Test compilation error handling."""
        mock_metadata = {"api_request_error": None, "api_status": "Failed", "compile_status": "Error", "compile_stderr": "SyntaxError: invalid syntax", "run_status": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-4, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print('Hello World'", timeout=30)
            assert "Compilation failed" in result
            assert "compile_stderr: SyntaxError: invalid syntax" in result
            assert success is False

    def test_compile_timeout(self, mock_tool):
        """Test compilation timeout handling."""
        mock_metadata = {"api_request_error": None, "api_status": "Failed", "compile_status": "TimeLimitExceeded", "compile_duration": 15.5, "run_status": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-4, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "complex_code", timeout=30)
            assert "Compilation time limit exceeded, time: 15.5" in result
            assert success is False

    def test_runtime_timeout(self, mock_tool):
        """Test runtime timeout handling."""
        mock_metadata = {"api_request_error": None, "api_status": "Failed", "compile_status": "Finished", "run_status": "TimeLimitExceeded", "duration": 30.0, "stdout": "Some partial output", "stderr": "Execution interrupted"}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-3, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "while True: pass", timeout=30)
            assert "Execution time limit exceeded, time: 30.0, timeout: 30" in result
            assert "stdout: Some partial output" in result
            assert "stderr: Execution interrupted" in result
            assert success is False

    def test_runtime_error(self, mock_tool):
        """Test runtime error handling."""
        mock_metadata = {"api_request_error": None, "api_status": "Failed", "compile_status": "Finished", "run_status": "Finished", "exit_code": 1, "stdout": "", "stderr": "ZeroDivisionError: division by zero"}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-2, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "x = 1/0", timeout=30)
            assert "return_code: 1" in result
            assert "stderr: ZeroDivisionError: division by zero" in result
            assert success is False

    def test_successful_execution_with_warnings(self, mock_tool):
        """Test successful execution that has warnings (stderr but exit_code=0)."""
        mock_metadata = {"api_request_error": None, "api_status": "Success", "run_status": "Finished", "exit_code": 0, "stdout": "42\n", "stderr": "DeprecationWarning: some warning"}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(True, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print(42)", timeout=30)
            assert "stdout: 42\n" in result
            assert "stderr: DeprecationWarning: some warning" in result
            assert success is False

    def test_unknown_api_status(self, mock_tool):
        """Test handling of unknown API status."""
        mock_metadata = {
            "api_request_error": None,
            "api_status": "UnknownStatus",
        }

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-1, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert "Unknown API status: UnknownStatus" in result
            assert success is False

    def test_unexpected_success_state(self, mock_tool):
        """Test handling of unexpected success state."""
        mock_metadata = {
            "api_request_error": None,
            "api_status": "Success",
            "run_status": "Unknown",
        }

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-1, mock_metadata)):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert "Unexpected execution state: Unknown" in result
            assert success is False

    def test_exception_handling_with_retry(self, mock_tool):
        """Test exception handling with retry mechanism."""
        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", side_effect=[Exception("Network error"), Exception("Still failing"), Exception("Final failure")]), patch("time.sleep"), patch("random.uniform", return_value=1.0):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert "Error in calling code interpreter: Final failure" in result
            assert success is False

    def test_max_retries_exceeded(self, mock_tool):
        """Test behavior when max retries are exceeded."""
        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", side_effect=Exception("Persistent error")), patch("time.sleep"), patch("random.uniform", return_value=1.0):
            result, success = mock_tool.execute_code("test_instance", "print('test')", timeout=30)
            assert "Error in calling code interpreter: Persistent error" in result
            assert success is False


class TestSandboxFusionToolComparison:
    """Test class to compare run_code and sim_jupyter modes."""

    @pytest.fixture
    def run_code_tool(self):
        """Create a run_code mode tool."""
        config = {"num_workers": 1, "rate_limit": 1, "default_timeout": 30, "enable_global_rate_limit": False, "sandbox_fusion_url": "http://mock-sandbox:8000", "mode": "run_code"}
        schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for execute code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "code needs to be execute"},
                        },
                        "required": ["code"],
                    },
                },
            }
        )
        with patch("verl.tools.sandbox_fusion_tools.init_execution_pool"):
            tool = SandboxFusionTool(config, schema)
            tool.execution_pool = Mock()
            return tool

    def test_error_message_consistency(self, run_code_tool):
        """Test that run_code mode now provides similar error detail as sim_jupyter mode."""
        # Test API error consistency
        mock_metadata = {"api_request_error": "Connection refused", "api_status": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-1, mock_metadata)):
            result, success = run_code_tool.execute_code("test_instance", "print('test')", timeout=30)
            # Should now provide detailed error message like sim_jupyter
            assert "Error in calling code interpreter" in result
            assert "Connection refused" in result
            assert success is False

    def test_timeout_message_format(self, run_code_tool):
        """Test that timeout messages follow sim_jupyter format."""
        mock_metadata = {"api_request_error": None, "api_status": "Failed", "compile_status": "Finished", "run_status": "TimeLimitExceeded", "duration": 25.0, "stdout": "partial output", "stderr": None}

        with patch("verl.utils.reward_score.sandbox_fusion.utils._process_single_case", return_value=(-3, mock_metadata)):
            result, success = run_code_tool.execute_code("test_instance", "long_running_code", timeout=30)
            # Should now provide detailed timeout info like sim_jupyter
            assert "Execution time limit exceeded" in result
            assert "time: 25.0" in result
            assert "timeout: 30" in result
            assert "stdout: partial output" in result
            assert success is False


@pytest.mark.asyncio
class TestSandboxFusionToolAsync:
    """Test async functionality of SandboxFusionTool."""

    @pytest.fixture
    async def async_tool(self):
        """Create a tool for async testing."""
        config = {"num_workers": 1, "rate_limit": 1, "default_timeout": 30, "enable_global_rate_limit": False, "sandbox_fusion_url": "http://mock-sandbox:8000", "mode": "run_code"}
        schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for execute code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "code needs to be execute"},
                        },
                        "required": ["code"],
                    },
                },
            }
        )
        with patch("verl.tools.sandbox_fusion_tools.init_execution_pool"):
            tool = SandboxFusionTool(config, schema)
            tool.execution_pool = AsyncMock()
            return tool

    async def test_async_execute_impl(self, async_tool):
        """Test the async _execute_impl method."""
        # Mock the execution pool to return a successful result

        async_tool.execution_pool.execute.remote.return_value = "Test output\n"

        result, reward, success, metrics = await async_tool._execute_impl("test_instance", {"code": "print('Test output')"})

        assert result == "Test output\n"
        assert success is True
        assert "code_length" in metrics
        assert metrics["execution_mode"] == "run_code"
