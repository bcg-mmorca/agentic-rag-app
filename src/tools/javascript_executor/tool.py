"""
Tool to execute javascript code.

To use this tool, you must have node.js installed in your machine.
"""

import json
import os
import subprocess
import tempfile

from typing import Any, List, Optional, Type
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class JSCodeExecutorInput(BaseModel):
    """Input for the JSCodeExecutor tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str = Field(
        description="Javascript code to execute"
    )
    args: List[any] = Field(
        description="List of arguments to pass to the JavaScript code"
    )

    # TODO: Add validation check if nodejs is installed on system


class JSCodeExecutor(BaseTool):  # type: ignore[override]
    """Tool that executes javascript code"""

    name: str = "js_code_executor"
    description: str = (
        """Executes JavaScript code in a Node.js environment with optional arguments.

        Args:
            code (str): JavaScript code to execute. Can be a single line or multiple lines
                of valid JavaScript code.
            args (List[any], optional): List of arguments to pass to the JavaScript code.
                These will be available in the JavaScript context as 'args' array.
                Defaults to None.

        Returns:
            str: The stdout output from the JavaScript code execution, stripped of
                leading/trailing whitespace.

        Examples:
            >>> execute_code('console.log("Hello, World!")')
            'Hello, World!'
            
            >>> execute_code('console.log(args[0])', ['Test'])
            'Test'
            
            >>> execute_code('console.log(args.reduce((a,b) => a+b, 0))', [1,2,3])
            '6'
        """
    )
    args_schema: Optional[Type[BaseModel]] = JSCodeExecutorInput

    def _run(
        self,
        code: str,
        args: List[any],
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        try:
            return self.execute(code, args)
        except Exception as e:
            return repr(e)

    @staticmethod
    def execute(code: str, args: List[any] = None) -> str:
        try:
            # Create a temporary JavaScript file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                # If args are provided, add them as JSON at the start of the file
                if args:
                    f.write(f'const args = {json.dumps(args)};\n')
                
                # Write the actual code
                f.write(code)
                temp_file = f.name

            # Execute the JavaScript file using Node.js
            result = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up the temporary file
            os.unlink(temp_file)
    
            return result.stdout.strip()
        
        except subprocess.CalledProcessError as e:
            return f"JavaScript execution error: {e.stderr}"
        except Exception as e:
            return f"Error: {str(e)}"
