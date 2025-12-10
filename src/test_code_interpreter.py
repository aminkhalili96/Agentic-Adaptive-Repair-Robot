"""
Tests for the Code Interpreter sandbox.

Tests security, execution, and return validation of LLM-generated path code.
"""

import pytest
import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.planning.code_interpreter import (
    exec_custom_path,
    validate_generated_code,
    CustomPathResult,
    EXAMPLE_STAR_CODE,
    EXAMPLE_HEXAGON_CODE,
    EXAMPLE_ZIGZAG_CODE,
)


class TestCodeValidation:
    """Test the pre-execution code validation."""
    
    def test_valid_code_passes(self):
        """Valid numpy code should pass validation."""
        code = '''
def generate_custom_path(center, radius):
    import numpy as np
    return np.array([[0, 0, 0], [1, 1, 1]])
'''
        is_safe, error = validate_generated_code(code)
        assert is_safe, f"Validation failed: {error}"
    
    def test_os_import_blocked(self):
        """Import os should be blocked."""
        code = '''
import os
def generate_custom_path(center, radius):
    return os.getcwd()
'''
        is_safe, error = validate_generated_code(code)
        assert not is_safe
        assert "os" in error.lower() or "import" in error.lower()
    
    def test_subprocess_blocked(self):
        """Subprocess import should be blocked."""
        code = '''
import subprocess
def generate_custom_path(center, radius):
    subprocess.run(['ls'])
'''
        is_safe, error = validate_generated_code(code)
        assert not is_safe
    
    def test_open_blocked(self):
        """File open() should be blocked."""
        code = '''
def generate_custom_path(center, radius):
    with open('/etc/passwd') as f:
        return f.read()
'''
        is_safe, error = validate_generated_code(code)
        assert not is_safe
        assert "open" in error.lower()
    
    def test_eval_blocked(self):
        """eval() should be blocked."""
        code = '''
def generate_custom_path(center, radius):
    return eval("1+1")
'''
        is_safe, error = validate_generated_code(code)
        assert not is_safe
    
    def test_dunder_access_blocked(self):
        """Dunder attribute access should be blocked."""
        code = '''
def generate_custom_path(center, radius):
    return ().__class__.__bases__
'''
        is_safe, error = validate_generated_code(code)
        assert not is_safe


class TestSandboxExecution:
    """Test the sandboxed execution of code."""
    
    def test_valid_star_pattern(self):
        """Example star code should execute successfully."""
        result = exec_custom_path(EXAMPLE_STAR_CODE, (0, 0, 0), 0.05)
        
        assert result.success, f"Execution failed: {result.error_message}"
        assert result.points is not None
        assert result.points.shape[1] == 3
        assert len(result.points) > 5  # Star should have multiple points
    
    def test_valid_hexagon_pattern(self):
        """Example hexagon code should execute successfully."""
        result = exec_custom_path(EXAMPLE_HEXAGON_CODE, (0.5, 0.5, 0.3), 0.08)
        
        assert result.success
        assert result.points is not None
        assert result.points.shape[1] == 3
        assert len(result.points) == 7  # 6 sides + 1 to close
    
    def test_valid_zigzag_pattern(self):
        """Example zigzag code should execute successfully."""
        result = exec_custom_path(EXAMPLE_ZIGZAG_CODE, (0, 0, 0), 0.1)
        
        assert result.success
        assert result.points is not None
        assert result.points.ndim == 2
        assert result.points.shape[1] == 3
    
    def test_simple_code_execution(self):
        """Simple valid code should work."""
        code = '''
def generate_custom_path(center, radius):
    import numpy as np
    cx, cy, cz = center
    return np.array([[cx, cy, cz], [cx + radius, cy, cz]])
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        
        assert result.success
        assert len(result.points) == 2
        assert result.points[1, 0] == 1.0  # cx + radius
    
    def test_missing_function_fails(self):
        """Code without generate_custom_path function should fail."""
        code = '''
def some_other_function(x):
    return x * 2
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        
        assert not result.success
        assert "generate_custom_path" in result.error_message
    
    def test_wrong_return_type_fails(self):
        """Returning wrong type should fail."""
        code = '''
def generate_custom_path(center, radius):
    return [1, 2, 3]  # List instead of numpy array
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        
        assert not result.success
        assert "numpy array" in result.error_message.lower()
    
    def test_wrong_shape_fails(self):
        """Returning wrong shape should fail."""
        code = '''
def generate_custom_path(center, radius):
    import numpy as np
    return np.array([1, 2, 3])  # 1D array instead of (N, 3)
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        
        assert not result.success
        assert "shape" in result.error_message.lower()
    
    def test_empty_result_fails(self):
        """Returning empty array should fail."""
        code = '''
def generate_custom_path(center, radius):
    import numpy as np
    return np.array([]).reshape(0, 3)
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        
        assert not result.success
        assert "at least one point" in result.error_message.lower()
    
    def test_blocked_import_in_execution(self):
        """Blocked imports should fail even if validation is bypassed."""
        code = '''
def generate_custom_path(center, radius):
    import sys  # Blocked!
    return None
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        
        assert not result.success


class TestSecurityBoundaries:
    """Test that security boundaries are enforced."""
    
    def test_cannot_access_builtins(self):
        """Cannot access dangerous builtins."""
        code = '''
def generate_custom_path(center, radius):
    __builtins__['open']('/etc/passwd')
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        assert not result.success
    
    def test_numpy_is_available(self):
        """numpy should be available."""
        code = '''
def generate_custom_path(center, radius):
    import numpy as np
    return np.zeros((5, 3))
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        assert result.success
        assert result.points.shape == (5, 3)
    
    def test_math_is_available(self):
        """math module should be available."""
        code = '''
def generate_custom_path(center, radius):
    import numpy as np
    import math
    x = math.sin(math.pi / 4)
    return np.array([[x, 0, 0]])
'''
        result = exec_custom_path(code, (0, 0, 0), 1.0)
        assert result.success


class TestResultObject:
    """Test the CustomPathResult dataclass."""
    
    def test_success_result(self):
        """Successful result has proper fields."""
        result = exec_custom_path(EXAMPLE_STAR_CODE, (0, 0, 0), 0.05)
        
        assert result.success is True
        assert result.error_message == ""
        assert result.generated_code == EXAMPLE_STAR_CODE
        
        d = result.to_dict()
        assert d["success"] is True
        assert d["num_points"] > 0
    
    def test_failure_result(self):
        """Failed result has proper error message."""
        code = 'invalid python code %%$$'
        result = exec_custom_path(code, (0, 0, 0), 0.05)
        
        assert result.success is False
        assert result.error_message != ""
        assert result.points is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
