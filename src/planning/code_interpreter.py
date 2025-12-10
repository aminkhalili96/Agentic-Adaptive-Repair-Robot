"""
Code Interpreter for Custom Path Generation.

Provides a safe sandbox execution environment for LLM-generated path code.
Enables the robot to demonstrate "General Intelligence" by writing its own
path generation algorithms for novel geometric patterns.

Security Model:
- Restricted builtins (no file I/O, no imports beyond numpy/math)
- AST-based code validation before execution
- Timeout protection (2 seconds default)
- Return type validation (must be numpy array of shape (N, 3))
"""

import ast
import signal
import traceback
from dataclasses import dataclass
from typing import Tuple, Optional, Set
import numpy as np
import math


# Keywords that are forbidden as function calls or attributes
# Note: We only check via AST, not text matching (to avoid false positives like 'os' in 'np.cos')
FORBIDDEN_NAMES: Set[str] = {
    'open',
    'file',
    'exec',
    'eval',
    'compile',
    '__import__',
    'subprocess',
    'shutil',
    'pathlib',
    'socket',
    'urllib',
    'requests',
    'pickle',
    'marshal',
    'ctypes',
    'multiprocessing',
    'threading',
    'globals',
    'locals',
    'vars',
    'dir',
    'getattr',
    'setattr',
    'delattr',
    'hasattr',
    '__builtins__',
    '__class__',
    '__bases__',
    '__subclasses__',
    '__mro__',
    '__code__',
}

# Forbidden module imports
FORBIDDEN_IMPORTS: Set[str] = {
    'os', 'sys', 'subprocess', 'shutil', 'pathlib',
    'socket', 'urllib', 'requests', 'pickle', 'marshal',
    'ctypes', 'multiprocessing', 'threading', 'builtins',
    'importlib', 'code', 'codeop', 'pty', 'fcntl', 'grp', 'pwd',
}

# Safe builtins allowed in the sandbox
SAFE_BUILTINS = {
    'True': True,
    'False': False,
    'None': None,
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'print': print,
    'range': range,
    'reversed': reversed,
    'round': round,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
}


# ============ RESULT DATACLASS ============

@dataclass
class CustomPathResult:
    """
    Result of custom path code execution.
    
    Attributes:
        points: numpy array of (x, y, z) waypoints, shape (N, 3)
        success: True if code executed successfully
        error_message: Error description if failed
        generated_code: The code that was executed
    """
    points: Optional[np.ndarray]
    success: bool
    error_message: str = ""
    generated_code: str = ""
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "num_points": len(self.points) if self.points is not None else 0,
            "generated_code": self.generated_code,
        }


# ============ TIMEOUT HANDLER ============

class TimeoutError(Exception):
    """Raised when code execution exceeds timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timed out (exceeded 2 seconds)")


# ============ CODE VALIDATION ============

class CodeSecurityValidator(ast.NodeVisitor):
    """
    AST visitor that checks for forbidden operations in code.
    """
    
    def __init__(self):
        self.violations = []
    
    def visit_Import(self, node):
        """Check import statements."""
        for alias in node.names:
            if alias.name not in ('numpy', 'math', 'np'):
                self.violations.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check from ... import statements."""
        if node.module not in ('numpy', 'math'):
            self.violations.append(f"Forbidden import from: {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                self.violations.append(f"Forbidden function: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_NAMES:
                self.violations.append(f"Forbidden attribute: {node.func.attr}")
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Check variable names for forbidden patterns."""
        if node.id.startswith('__') and node.id.endswith('__'):
            if node.id not in ('__name__', '__doc__'):
                self.violations.append(f"Forbidden dunder access: {node.id}")
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Check attribute access for forbidden patterns."""
        if node.attr.startswith('__') and node.attr.endswith('__'):
            self.violations.append(f"Forbidden dunder attribute: {node.attr}")
        if node.attr in FORBIDDEN_NAMES:
            self.violations.append(f"Forbidden attribute: {node.attr}")
        self.generic_visit(node)


def validate_generated_code(code_str: str) -> Tuple[bool, str]:
    """
    Validate generated code for security issues.
    
    Args:
        code_str: Python code string to validate
        
    Returns:
        Tuple of (is_safe, error_message)
    """
    # Parse and validate AST (no text-based checks to avoid false positives)
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    
    validator = CodeSecurityValidator()
    validator.visit(tree)
    
    if validator.violations:
        return False, f"SecurityError: {'; '.join(validator.violations)}"
    
    return True, ""


# ============ SANDBOX EXECUTION ============

def exec_custom_path(
    code_str: str,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.05,
    timeout_seconds: int = 2
) -> CustomPathResult:
    """
    Execute custom path generation code in a sandboxed environment.
    
    The code must define a function `generate_custom_path(center, radius)`
    that returns a numpy array of shape (N, 3) containing (x, y, z) points.
    
    Args:
        code_str: Python code containing generate_custom_path function
        center: Center point (x, y, z) for the path
        radius: Radius/size parameter for the path
        timeout_seconds: Maximum execution time (default 2s)
        
    Returns:
        CustomPathResult with points array or error message
        
    Security:
        - Only numpy and math are available
        - No file I/O or system access
        - Timeout protection
        - Return type validation
    """
    # Validate code before execution
    is_safe, error_msg = validate_generated_code(code_str)
    if not is_safe:
        return CustomPathResult(
            points=None,
            success=False,
            error_message=error_msg,
            generated_code=code_str
        )
    
    # Create a safe import function that only allows numpy and math
    def safe_import(name, *args, **kwargs):
        if name in ('numpy', 'np'):
            return np
        elif name == 'math':
            return math
        else:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    
    # Prepare safe execution environment
    safe_globals = {
        '__builtins__': {**SAFE_BUILTINS, '__import__': safe_import},
        'np': np,
        'numpy': np,
        'math': math,
        'pi': math.pi,
        'sin': math.sin,
        'cos': math.cos,
        'sqrt': math.sqrt,
    }
    safe_locals = {}
    
    # Set timeout (Unix only - Windows will skip)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    except (AttributeError, ValueError):
        # Windows doesn't support SIGALRM
        pass
    
    try:
        # Execute the code to define the function
        exec(code_str, safe_globals, safe_locals)
        
        # Check that the function was defined
        if 'generate_custom_path' not in safe_locals:
            return CustomPathResult(
                points=None,
                success=False,
                error_message="Code must define function 'generate_custom_path(center, radius)'",
                generated_code=code_str
            )
        
        # Call the function
        func = safe_locals['generate_custom_path']
        result = func(center, radius)
        
        # Cancel timeout
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        
        # Validate result
        if not isinstance(result, np.ndarray):
            return CustomPathResult(
                points=None,
                success=False,
                error_message=f"Return type must be numpy array, got {type(result).__name__}",
                generated_code=code_str
            )
        
        if result.ndim != 2 or result.shape[1] != 3:
            return CustomPathResult(
                points=None,
                success=False,
                error_message=f"Return shape must be (N, 3), got {result.shape}",
                generated_code=code_str
            )
        
        if len(result) == 0:
            return CustomPathResult(
                points=None,
                success=False,
                error_message="Path must contain at least one point",
                generated_code=code_str
            )
        
        return CustomPathResult(
            points=result,
            success=True,
            generated_code=code_str
        )
        
    except TimeoutError as e:
        return CustomPathResult(
            points=None,
            success=False,
            error_message=str(e),
            generated_code=code_str
        )
    except Exception as e:
        return CustomPathResult(
            points=None,
            success=False,
            error_message=f"ExecutionError: {str(e)}",
            generated_code=code_str
        )
    finally:
        # Restore old signal handler
        if old_handler is not None:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass


# ============ EXAMPLE PATTERNS ============

EXAMPLE_STAR_CODE = '''
def generate_custom_path(center, radius):
    """Generate a 5-pointed star pattern."""
    import numpy as np
    
    cx, cy, cz = center
    points = []
    num_points = 5
    inner_radius = radius * 0.4
    
    for i in range(num_points * 2):
        angle = (i * np.pi / num_points) - np.pi / 2
        if i % 2 == 0:
            r = radius  # Outer point
        else:
            r = inner_radius  # Inner point
        
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        z = cz
        points.append([x, y, z])
    
    # Close the star
    points.append(points[0])
    
    return np.array(points)
'''

EXAMPLE_HEXAGON_CODE = '''
def generate_custom_path(center, radius):
    """Generate a hexagonal pattern."""
    import numpy as np
    
    cx, cy, cz = center
    points = []
    num_sides = 6
    
    for i in range(num_sides + 1):
        angle = (i * 2 * np.pi / num_sides) - np.pi / 2
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        z = cz
        points.append([x, y, z])
    
    return np.array(points)
'''

EXAMPLE_ZIGZAG_CODE = '''
def generate_custom_path(center, radius):
    """Generate a zigzag pattern."""
    import numpy as np
    
    cx, cy, cz = center
    points = []
    num_zigs = 8
    amplitude = radius
    
    for i in range(num_zigs + 1):
        x = cx - radius + (2 * radius * i / num_zigs)
        y = cy + (amplitude if i % 2 == 0 else -amplitude) * 0.5
        z = cz
        points.append([x, y, z])
    
    return np.array(points)
'''


def get_example_code(pattern_name: str) -> str:
    """
    Get example code for a named pattern.
    
    Args:
        pattern_name: Name like 'star', 'hexagon', 'zigzag'
        
    Returns:
        Example code string or empty string if not found
    """
    patterns = {
        'star': EXAMPLE_STAR_CODE,
        'hexagon': EXAMPLE_HEXAGON_CODE,
        'zigzag': EXAMPLE_ZIGZAG_CODE,
    }
    return patterns.get(pattern_name.lower(), '')
