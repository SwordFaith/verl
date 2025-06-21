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


@pytest.fixture
def tool_schema():
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
def tool_config():
    return {"sandbox_fusion_url": "http://test-sandbox.example.com", "default_timeout": 30, "cell_timeout": 10, "num_workers": 2, "rate_limit": 5, "mode": "run_code"}


@pytest.fixture
def sandbox_tool(tool_config, tool_schema):
    with patch("verl.tools.sandbox_fusion_tools.init_execution_pool"):
        tool = SandboxFusionTool(tool_config, tool_schema)
        tool.tool_logger = Mock()
        # Mock the execution pool with AsyncMock
        tool.execution_pool = Mock()
        tool.execution_pool.execute = AsyncMock()
        return tool


class TestSandboxFusionTool:
    """Test suite for SandboxFusionTool covering all execution modes"""

    @pytest.mark.asyncio
    async def test_create_instance(self, sandbox_tool):
        """Test instance creation"""
        instance_id = await sandbox_tool.create(ground_truth="test")
        assert instance_id in sandbox_tool._instance_dict
        assert sandbox_tool._instance_dict[instance_id]["ground_truth"] == "test"
        assert sandbox_tool._instance_dict[instance_id]["cells"] == []

    @pytest.mark.asyncio
    async def test_empty_code_handling(self, sandbox_tool):
        """Test handling of empty code input"""
        instance_id = await sandbox_tool.create()

        # Test empty code
        result, reward, success, metrics = await sandbox_tool._execute_impl(instance_id, {"code": ""})

        assert result == "no code parsed"
        assert not success
        assert metrics["error_type"] == "empty_code"
        assert metrics["lines_of_code"] == 0
        assert metrics["execution_mode"] == "run_code"

    def test_format_execution_result_success(self, sandbox_tool):
        """Test clean formatting for successful execution"""
        # Test with stdout
        api_metadata = {"stdout": "Hello World\n"}
        result = sandbox_tool._format_execution_result("", True, api_metadata)
        assert result == "Hello World"

        # Test without stdout but with response
        result = sandbox_tool._format_execution_result("42", True, {})
        assert result == "42"

        # Test with no output
        result = sandbox_tool._format_execution_result("", True, {})
        assert result == "Execution completed successfully"

    def test_format_execution_result_error(self, sandbox_tool):
        """Test clean formatting for error cases with suggestions"""
        # Test NameError with suggestion
        api_metadata = {"stderr": "NameError: name 'x' is not defined"}
        result = sandbox_tool._format_execution_result("", False, api_metadata)
        assert "NameError: name 'x' is not defined" in result
        assert "Suggestion: Check variable name" in result

        # Test generic error without suggestion
        api_metadata = {"stderr": "Some generic error"}
        result = sandbox_tool._format_execution_result("", False, api_metadata)
        assert result == "Some generic error"

    def test_python_error_suggestions(self, sandbox_tool):
        """Test all Python error suggestion patterns"""
        test_cases = [
            ("NameError: name 'x' is not defined", "Check variable name"),
            ("TypeError: unsupported operand", "Verify data types"),
            ("ZeroDivisionError: division by zero", "Add check for division"),
            ("SyntaxError: invalid syntax", "Check Python syntax"),
            ("IndentationError: expected indent", "Fix code indentation"),
            ("ModuleNotFoundError: No module named", "Check if required module"),
            ("ImportError: cannot import", "Verify module installation"),
            ("AttributeError: object has no attribute", "Check object attributes"),
            ("KeyError: 'missing_key'", "Verify dictionary keys"),
            ("IndexError: list index out of range", "Check list/array bounds"),
            ("ValueError: invalid literal", "Check input values"),
        ]

        for error_msg, expected_suggestion in test_cases:
            suggestion = sandbox_tool._get_python_error_suggestion(error_msg)
            assert expected_suggestion in suggestion

    def test_error_classification(self, sandbox_tool):
        """Test fine-grained error classification"""
        test_cases = [
            ("NameError: name 'x' is not defined", "unknown", "name_error"),
            ("TypeError: unsupported operand", "unknown", "type_error"),
            ("ZeroDivisionError: division by zero", "unknown", "zero_division_error"),
            ("SyntaxError: invalid syntax", "unknown", "syntax_error"),
            ("IndentationError: expected indent", "unknown", "indentation_error"),
            ("ModuleNotFoundError: No module", "unknown", "module_not_found_error"),
            ("ImportError: cannot import", "unknown", "import_error"),
            ("AttributeError: no attribute", "unknown", "attribute_error"),
            ("KeyError: 'key'", "unknown", "key_error"),
            ("IndexError: out of range", "unknown", "index_error"),
            ("ValueError: invalid", "unknown", "value_error"),
            ("FileNotFoundError: file not found", "unknown", "file_not_found_error"),
            ("PermissionError: access denied", "unknown", "permission_error"),
            ("MemoryError: out of memory", "unknown", "memory_error"),
            ("RecursionError: maximum recursion", "unknown", "recursion_error"),
            ("KeyboardInterrupt", "unknown", "keyboard_interrupt"),
            ("Gateway Timeout", "unknown", "api_timeout_error"),
            ("time limit exceeded", "unknown", "timeout_error"),
            ("sandbox error", "SandboxError", "sandbox_error"),  # Fixed: non-empty error message
            ("compilation failed", "unknown", "compilation_error"),
            ("some generic error", "unknown", "runtime_error"),
            ("execution stopped", "unknown", "execution_failure"),
        ]

        for error_msg, api_status, expected_type in test_cases:
            error_type = sandbox_tool._classify_error_type(error_msg, api_status)
            assert error_type == expected_type, f"Failed for: {error_msg}"

    def test_execute_code_success(self, sandbox_tool):
        """Test successful run_code execution"""
        # Mock successful execution directly on the method
        with patch.object(sandbox_tool, "execute_code") as mock_execute:
            mock_execute.return_value = ("Hello World", True, {"api_execution_time_ms": 1500, "api_return_code": 0, "api_execution_status": "Finished", "has_stdout": True, "has_stderr": False})

            result, success, metrics = sandbox_tool.execute_code("test_id", "print('Hello World')", 30, "python")

            assert result == "Hello World"
            assert success
            assert metrics["api_execution_time_ms"] == 1500
            assert metrics["api_return_code"] == 0
            assert metrics["has_stdout"]

    def test_execute_code_runtime_error(self, sandbox_tool):
        """Test run_code execution with runtime error"""
        with patch.object(sandbox_tool, "execute_code") as mock_execute:
            mock_execute.return_value = ("NameError: name 'x' is not defined\nSuggestion: Check variable name and ensure it's defined before use", False, {"api_execution_time_ms": 500, "api_return_code": 1, "api_execution_status": "Error", "has_stdout": False, "has_stderr": True})

            result, success, metrics = sandbox_tool.execute_code("test_id", "print(x)", 30, "python")

            assert "NameError: name 'x' is not defined" in result
            assert "Suggestion: Check variable name" in result
            assert not success
            assert metrics["api_execution_time_ms"] == 500
            assert metrics["api_return_code"] == 1

    def test_execute_code_timeout(self, sandbox_tool):
        """Test run_code execution timeout"""
        with patch.object(sandbox_tool, "execute_code") as mock_execute:
            mock_execute.return_value = ("Execution time limit exceeded, time: 30.0, timeout: 30", False, {"api_execution_time_ms": 30000, "api_execution_status": "TimeLimitExceeded", "has_timeout": True, "has_stdout": True, "has_stderr": False})

            result, success, metrics = sandbox_tool.execute_code("test_id", "while True: pass", 30, "python")

            assert "Execution time limit exceeded" in result
            assert not success
            assert metrics["has_timeout"]
            assert metrics["api_execution_time_ms"] == 30000

    @patch("requests.request")
    def test_get_jupyter_mode_result_success(self, mock_request, sandbox_tool):
        """Test successful run_jupyter execution"""
        # Configure tool for jupyter mode
        sandbox_tool.mode = "run_jupyter"
        sandbox_tool._instance_dict["test_id"] = {"cells": ["print('Hello')"]}

        # Mock successful jupyter response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Success", "driver": {"status": "Finished", "execution_time": 2.0}, "cells": [{"stdout": "Hello", "stderr": "", "display": [], "error": []}]}
        mock_request.return_value = mock_response

        result, success, metrics = sandbox_tool.get_jupyter_mode_result("test_id", 30)

        assert result == "Hello"
        assert success
        assert metrics["api_execution_time_ms"] == 2000
        assert metrics["display_object_count"] == 0
        assert metrics["has_stdout"]

    @patch("requests.request")
    def test_get_jupyter_mode_result_with_errors(self, mock_request, sandbox_tool):
        """Test run_jupyter execution with cell errors"""
        sandbox_tool.mode = "run_jupyter"
        sandbox_tool._instance_dict["test_id"] = {"cells": ["print(x)"]}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Success", "driver": {"status": "Finished", "execution_time": 1.0}, "cells": [{"stdout": "", "stderr": "NameError: name 'x' is not defined", "display": [], "error": [{"type": "NameError", "message": "name 'x' is not defined"}]}]}
        mock_request.return_value = mock_response

        result, success, metrics = sandbox_tool.get_jupyter_mode_result("test_id", 30)

        assert "NameError: name 'x' is not defined" in result
        assert "Suggestion: Check variable name" in result
        assert not success
        assert metrics["has_cell_errors"]

    @patch("requests.request")
    def test_get_sim_jupyter_mode_result_success(self, mock_request, sandbox_tool):
        """Test successful sim_jupyter execution"""
        sandbox_tool.mode = "sim_jupyter"
        sandbox_tool._instance_dict["test_id"] = {"cells": ["result = 2 + 2", "result"]}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Success", "run_result": {"status": "Finished", "return_code": 0, "execution_time": 0.8, "stdout": "'result': 4", "stderr": ""}}
        mock_request.return_value = mock_response

        result, success, metrics = sandbox_tool.get_sim_jupyter_mode_result("test_id", 30)

        assert result == "'result': 4"
        assert success
        assert metrics["api_execution_time_ms"] == 800
        assert metrics["api_return_code"] == 0

    @patch("requests.request")
    def test_get_sim_jupyter_mode_result_timeout(self, mock_request, sandbox_tool):
        """Test sim_jupyter execution timeout"""
        sandbox_tool.mode = "sim_jupyter"
        sandbox_tool._instance_dict["test_id"] = {"cells": ["while True: pass"]}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Failed", "run_result": {"status": "TimeLimitExceeded", "return_code": None, "execution_time": 30.0, "stdout": "", "stderr": ""}}
        mock_request.return_value = mock_response

        result, success, metrics = sandbox_tool.get_sim_jupyter_mode_result("test_id", 30)

        assert "Execution time limit exceeded" in result
        assert not success
        assert metrics["has_timeout"]
        assert len(sandbox_tool._instance_dict["test_id"]["cells"]) == 0  # Cell should be dropped

    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, sandbox_tool):
        """Test complete execution flow"""
        # Create instance
        instance_id = await sandbox_tool.create()

        # Mock execution pool for end-to-end test
        sandbox_tool.execution_pool.execute.remote.return_value = ("42", True, {"api_execution_time_ms": 500, "api_return_code": 0, "api_execution_status": "Finished", "has_stdout": True, "has_stderr": False})

        # Execute code
        result, reward, success, metrics = await sandbox_tool._execute_impl(instance_id, {"code": "print(6 * 7)"})

        # Verify results
        assert result == "42"
        assert success
        assert metrics["lines_of_code"] == 1
        assert metrics["execution_time_ms"] == 500
        assert metrics["execution_mode"] == "run_code"
        assert metrics["error_type"] is None

    @pytest.mark.asyncio
    async def test_release_instance(self, sandbox_tool):
        """Test instance cleanup"""
        instance_id = await sandbox_tool.create()
        assert instance_id in sandbox_tool._instance_dict

        await sandbox_tool.release(instance_id)
        assert instance_id not in sandbox_tool._instance_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
