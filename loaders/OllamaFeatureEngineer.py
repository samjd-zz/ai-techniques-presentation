"""
Ollama-Powered Feature Engineering Module

This module provides an intelligent feature engineering system that leverages
locally-running Large Language Models (LLMs) via Ollama to dynamically analyze 
datasets and suggest meaningful feature transformations. Unlike cloud-based 
solutions, this runs entirely on your local machine with no API costs.

Supported Models (must be installed via `ollama pull <model>`):
- llama3.1:8b (Balanced quality and speed - DEFAULT, 8B parameters)
- llama3.2:latest (Faster but less capable, 3B parameters)
- mistral:latest (Good alternative, 7B parameters)
- qwen2.5-coder:latest (Optimized for code, 7B parameters)
- deepseek-coder-v2:latest (Best quality, 16B parameters)

Key Features:
- No API costs - runs locally via Ollama
- Privacy-preserving - data never leaves your machine
- Automated dataset analysis and summary generation
- LLM-powered feature suggestions
- Safe execution with error handling
- Iterative feature engineering (multiple rounds of enhancement)
- Detailed explanations of generated features

Requirements:
- Ollama installed: https://ollama.ai/download
- At least one model pulled: `ollama pull llama3.2`
- Sufficient RAM (8GB+ recommended for 7B models)

Author: 80% claude-sonnet-4-5-20250929.ai  - 20% Shawn Jackson Dyck
Date: October 2025
"""

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations and array handling
import ollama  # Ollama client for local LLM interactions
import json  # JSON parsing for LLM responses
import re  # Regular expressions for extracting JSON from text
from typing import Dict, List, Any, Optional  # Type hints for better code documentation
import logging  # Logging for metrics and debugging
import time  # For timing operations
from datetime import datetime  # For timestamps
import argparse  # Command-line argument parsing


# ANSI Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'


# Unicode icons for better visual feedback
class Icons:
    """Unicode icons for visual feedback"""
    SUCCESS = '✓'
    ERROR = '✗'
    WARNING = '⚠'
    INFO = 'ℹ'
    ARROW_RIGHT = '→'
    CLOCK = '⏱'
    ROCKET = '🚀'
    BRAIN = '🧠'
    WRENCH = '🔧'
    CHECK_MARK = '✅'
    CROSS_MARK = '❌'
    HOURGLASS = '⏳'


class OllamaFeatureEngineer:
    """
    Intelligent feature engineering using locally-running Ollama LLMs.
    
    This class analyzes your dataset, generates feature engineering suggestions
    using a local LLM via Ollama, and applies them to create new predictive 
    features. It handles the entire workflow from data analysis to feature 
    creation with built-in error handling and safety checks.
    
    The LLM analyzes:
    - Data types (numeric, categorical, temporal)
    - Statistical properties (ranges, distributions)
    - Missing values and data quality issues
    - Target variable characteristics (if provided)
    
    It then suggests engineered features like:
    - Interaction terms (multiplicative combinations)
    - Polynomial transformations (quadratic, cubic terms)
    - Ratio features (feature1 / feature2)
    - Statistical transformations (log, sqrt)
    - Domain-specific features
    
    Attributes:
        model (str): The Ollama model to use
        generated_features (List[Dict]): List of features created by the LLM
        ollama_options (Dict): Additional options for Ollama (temperature, etc.)
    """
    
    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.2):
        """
        Initialize the Ollama Feature Engineer.
        
        Args:
            model (str, optional): Which Ollama model to use. Default is llama3.1:8b.
                                  Popular choices:
                                  - "llama3.1:8b": Balanced quality and speed (DEFAULT)
                                  - "llama3.2:latest": Faster, less capable (3B)
                                  - "mistral:latest": Good alternative (7B)
                                  - "qwen2.5-coder:latest": Code-focused (7B)
                                  - "deepseek-coder-v2:latest": Best quality (16B)
                                  
                                  Ensure the model is downloaded first:
                                  `ollama pull <model_name>`
            
            temperature (float, optional): LLM temperature for response creativity.
                                          Lower = more deterministic (0.0-1.0)
                                          Default: 0.2 (very focused, minimal creativity)
        
        Returns:
            None
        
        Raises:
            ImportError: If ollama package is not installed (`pip install ollama`)
            ConnectionError: If Ollama service is not running (`ollama serve`)
            
        Example:
            >>> # Start Ollama service first: ollama serve
            >>> # Pull a model: ollama pull llama3.1:8b
            >>> engineer = OllamaFeatureEngineer(model="llama3.1:8b")
        """
        self.model = model  # Store model name for later API calls
        self.generated_features = []  # Track features created in this session
        
        # Ollama generation options - optimized for llama3.1:8b
        # temperature: 0.2 = very focused, minimal creativity for consistent JSON
        # num_predict: 8000 = increased from 4000 to handle longer feature lists
        # top_p: 0.9 = nucleus sampling for better quality
        # repeat_penalty: 1.1 = slight penalty to avoid repetition
        self.ollama_options = {
            'temperature': temperature,
            'num_predict': -1,  # Increased for complete JSON responses
            'top_p': 0.9,
            'repeat_penalty': 1.1
        }
        
        # Initialize metrics tracking
        self.metrics = {
            'session_start': datetime.now(),
            'llm_calls': 0,
            'llm_call_times': [],
            'total_llm_time': 0.0,
            'json_parse_attempts': 0,
            'json_parse_failures': 0,
            'json_repairs_attempted': 0,
            'json_repairs_successful': 0,
            'features_suggested': 0,
            'features_created': 0,
            'features_failed': 0,
            'iterations_completed': 0,
            'errors': []
        }
        
        # Set up logging with custom colored formatter
        self.logger = logging.getLogger(f'OllamaFeatureEngineer_{id(self)}')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Custom formatter with colors
            class ColoredFormatter(logging.Formatter):
                """Custom formatter that adds colors based on log level"""
                
                LEVEL_COLORS = {
                    'DEBUG': Colors.CYAN,
                    'INFO': Colors.BRIGHT_GREEN,
                    'WARNING': Colors.BRIGHT_YELLOW,
                    'ERROR': Colors.BRIGHT_RED,
                    'CRITICAL': Colors.BRIGHT_RED + Colors.BOLD
                }
                
                LEVEL_ICONS = {
                    'DEBUG': Icons.INFO,
                    'INFO': Icons.INFO,
                    'WARNING': Icons.WARNING,
                    'ERROR': Icons.ERROR,
                    'CRITICAL': Icons.CROSS_MARK
                }
                
                def format(self, record):
                    # Get color and icon for this level
                    color = self.LEVEL_COLORS.get(record.levelname, Colors.RESET)
                    icon = self.LEVEL_ICONS.get(record.levelname, '')
                    
                    # Format the message with colors
                    levelname_colored = f"{color}{icon} {record.levelname}{Colors.RESET}"
                    
                    # Create formatted record
                    record.levelname_colored = levelname_colored
                    
                    # Use custom format
                    formatter = logging.Formatter(
                        '%(asctime)s - %(levelname_colored)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                    return formatter.format(record)
            
            console_handler.setFormatter(ColoredFormatter())
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"{Icons.ROCKET} {Colors.BRIGHT_CYAN}Initialized OllamaFeatureEngineer{Colors.RESET} with model: {Colors.BRIGHT_BLUE}{model}{Colors.RESET}")
        self.logger.info(f"{Icons.WRENCH} Temperature: {Colors.YELLOW}{temperature}{Colors.RESET}, Token limit: {Colors.YELLOW}{self.ollama_options['num_predict']}{Colors.RESET}")
        
        # Track if model is loaded
        self._model_loaded = False
        
        # Verify Ollama is available and model exists
        try:
            # Test connection to Ollama
            ollama.list()
        except Exception as e:
            raise ConnectionError(
                "Cannot connect to Ollama. Please ensure:\n"
                "1. Ollama is installed (https://ollama.ai/download)\n"
                "2. Ollama service is running (run 'ollama serve')\n"
                f"Error: {str(e)}"
            )
    
    def _unload_model(self) -> None:
        """
        Unload the model from Ollama to free up memory.
        
        This is called automatically on errors or when cleanup is needed.
        Ollama keeps models in memory for faster subsequent calls, but this
        can consume significant RAM. Unloading helps manage memory usage.
        
        Returns:
            None
        """
        try:
            if self._model_loaded:
                # Check if model is running
                running_models = ollama.ps()
                model_running = any(
                    m.get('model', '').startswith(self.model.split(':')[0]) 
                    for m in running_models.get('models', [])
                )
                
                if model_running:
                    self.logger.info(f"Unloading model {self.model} from memory")
                    # Unload by sending a request with keep_alive=0
                    ollama.generate(model=self.model, prompt="", keep_alive=0)
                    self._model_loaded = False
                    self.logger.info(f"Model {self.model} unloaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to unload model: {e}")
    
    def __del__(self):
        """Destructor to ensure model is unloaded when object is destroyed."""
        self._unload_model()
    
    def analyze_dataset(self, df: pd.DataFrame, target: str = None) -> str:
        """
        Generate a comprehensive dataset summary for the LLM.
        
        This method creates a structured text summary of your dataset including:
        - Overall shape (rows and columns)
        - Data types for each column
        - Missing value percentages
        - Cardinality (unique value counts)
        - Statistical summaries (for numeric columns)
        - Value distributions (for categorical columns)
        
        The LLM uses this summary to understand data characteristics and
        suggest appropriate feature transformations.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            target (str, optional): Name of the target variable for supervised learning.
                                   If provided, included in the summary so the LLM
                                   understands the prediction task.
        
        Returns:
            str: A formatted text summary of the dataset suitable for LLM input.
                 Includes statistics, data types, and missing value information.
        
        Example:
            >>> engineer = OllamaFeatureEngineer()
            >>> summary = engineer.analyze_dataset(df, target="price")
            >>> print(summary)
            Dataset Shape: (1000, 15)
            Column Information:
            - age: int64, 50 unique values, 0.0% missing
              Stats: min=18.00, max=80.00, mean=42.30
            ...
        """
        # Create header with dataset dimensions
        summary = f"Dataset Shape: {df.shape}\n\n"
        summary += "Column Information:\n"
        
        # Analyze each column in the dataset
        for col in df.columns:
            dtype = df[col].dtype  # Get data type
            null_pct = df[col].isnull().sum() / len(df) * 100  # Calculate missing % 
            unique = df[col].nunique()  # Count unique values
            
            # Add basic column information
            summary += f"- {col}: {dtype}, {unique} unique values, {null_pct:.1f}% missing\n"
            
            # For numeric columns: add statistical summary
            if df[col].dtype in ['int64', 'float64']:
                summary += f"  Stats: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"
            
            # For categorical columns (<=10 unique values): show distribution
            elif unique <= 10:
                summary += f"  Values: {df[col].value_counts().head().to_dict()}\n"
        
        # Include target variable information if provided
        if target:
            summary += f"\nTarget Variable: {target}\n"
        
        return summary
    
    def _validate_suggestions(self, suggestions: List[Dict[str, str]]) -> bool:
        """
        Validate that suggestions have the required structure.
        
        Args:
            suggestions: List of suggestion dictionaries to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(suggestions, list) or len(suggestions) == 0:
            return False
        
        required_keys = {'feature_name', 'code', 'explanation'}
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                return False
            if not required_keys.issubset(suggestion.keys()):
                return False
            # Check that values are non-empty strings
            for key in required_keys:
                if not isinstance(suggestion[key], str) or not suggestion[key].strip():
                    return False
        
        return True
    
    def _ask_llm_to_fix_json(self, broken_json: str, error_message: str) -> str:
        """
        Ask the LLM to fix malformed JSON.
        
        Args:
            broken_json: The malformed JSON string
            error_message: The error message from JSON parser
            
        Returns:
            str: Fixed JSON string
        """
        repair_prompt = f"""The following JSON is malformed and caused this error:
{error_message}

Broken JSON:
{broken_json}

Please fix this JSON and return ONLY the corrected JSON array with no other text. The JSON must be valid and complete. Each object must have "feature_name", "code", and "explanation" fields, and all strings must be properly quoted and closed."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': repair_prompt}],
                options=self.ollama_options,
                format='json'
            )
            return response['message']['content']
        except Exception as e:
            print(f"Failed to get repair response: {e}")
            raise
    
    def get_feature_suggestions(self, df: pd.DataFrame, target: str = None, 
                                task_type: str = None, 
                                max_retries: int = 3) -> List[Dict[str, str]]:
        """
        Ask Ollama LLM for feature engineering suggestions.
        
        This method sends the dataset summary to a local LLM via Ollama along 
        with task context and receives back 5-10 feature engineering suggestions. 
        Each suggestion includes:
        1. A descriptive feature name
        2. Executable Python code (pandas operations)
        3. Explanation of why the feature might be useful
        
        The LLM ensures all suggested code:
        - Uses only columns that exist in the dataset
        - Handles edge cases (division by zero, null values)
        - Returns valid pandas/numpy operations
        
        Args:
            df (pd.DataFrame): The dataset to generate features for
            target (str, optional): The target variable (used for context).
                                   Helps the LLM suggest relevant features.
                                   Example: "price" for regression, "churn" for classification
            task_type (str, optional): Type of ML task ("classification" or "regression").
                                      If None, will auto-detect based on target column dtype.
                                      Default: None (auto-detect)
                                      Helps LLM prioritize feature types
        
        Returns:
            List[Dict[str, str]]: List of feature suggestions, each containing:
                - "feature_name": Name for the new column (str)
                - "code": Pandas code to create the feature (str)
                - "explanation": Why this feature is useful (str)
        
        Raises:
            ValueError: If LLM response doesn't contain valid JSON
            ConnectionError: If Ollama service is not accessible
        
        Example:
            >>> suggestions = engineer.get_feature_suggestions(
            ...     df, 
            ...     target="price",
            ...     task_type="regression"
            ... )
            >>> for s in suggestions:
            ...     print(f"{s['feature_name']}: {s['explanation']}")
        """
        
        # Generate dataset summary for LLM context
        dataset_summary = self.analyze_dataset(df, target)
        
        # Auto-detect task_type if not provided
        if task_type is None and target is not None and target in df.columns:
            target_dtype = df[target].dtype
            
            # Determine task type based on target column's data type
            if target_dtype in ['int64', 'float64']:
                # Numeric columns typically indicate regression
                # However, if unique values are very few, might be classification
                unique_count = df[target].nunique()
                if unique_count <= 10:
                    task_type = "classification"
                    self.logger.info(f"Auto-detected task_type='classification' "
                                   f"(target '{target}' is numeric but has only {unique_count} unique values)")
                else:
                    task_type = "regression"
                    self.logger.info(f"Auto-detected task_type='regression' "
                                   f"(target '{target}' is numeric with {unique_count} unique values)")
            else:
                # Categorical, object, or boolean columns indicate classification
                task_type = "classification"
                unique_count = df[target].nunique()
                self.logger.info(f"Auto-detected task_type='classification' "
                               f"(target '{target}' is categorical with {unique_count} classes)")
        elif task_type is None:
            # Default to classification if we can't determine
            task_type = "classification"
            self.logger.warning("task_type not provided and could not auto-detect. Defaulting to 'classification'")
        else:
            # User provided task_type explicitly
            self.logger.info(f"Using user-provided task_type='{task_type}'")
        
        # Create the prompt for the LLM
        # This prompt is carefully structured to ensure:
        # 1. LLM understands the data characteristics
        # 2. LLM knows the prediction task
        # 3. LLM returns strictly formatted JSON
        prompt = f"""You are a machine learning expert. Analyze this dataset and suggest 5-10 useful feature engineering transformations.

{dataset_summary}

Task Type: {task_type}

For each suggestion, provide:
1. A descriptive name for the new feature
2. Python code to create it (using pandas operations on 'df')
3. Brief explanation of why it might be useful

Format your response as a JSON array like this:
[
  {{
    "feature_name": "name_of_feature",
    "code": "df['new_col'] = df['col1'] * df['col2']",
    "explanation": "why this helps"
  }}
]

IMPORTANT: 
- Code must be valid pandas operations
- Use only columns that exist in the dataset
- Handle potential errors (division by zero, null values, etc.)
- Return ONLY the JSON array, no other text before or after
- Ensure all JSON strings are properly closed with quotes"""

        # SELF-HEALING RETRY LOOP
        for attempt in range(max_retries):
            try:
                # Track LLM call metrics
                start_time = time.time()
                self.metrics['llm_calls'] += 1
                self.logger.info(f"{Icons.BRAIN} {Colors.BRIGHT_MAGENTA}LLM call #{self.metrics['llm_calls']}{Colors.RESET} - Attempt {Colors.YELLOW}{attempt + 1}/{max_retries}{Colors.RESET}")
                
                # Call Ollama API
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options=self.ollama_options,
                    format='json'
                )
                
                # Record timing
                call_time = time.time() - start_time
                self.metrics['llm_call_times'].append(call_time)
                self.metrics['total_llm_time'] += call_time
                self.logger.info(f"LLM responded in {call_time:.2f}s")
                
                # Mark model as loaded after successful call
                self._model_loaded = True
                
                # Extract the response content
                content = response['message']['content']
                
                # Extract JSON from response using regex
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                
                if not json_match:
                    if attempt < max_retries - 1:
                        print(f"⚠ Attempt {attempt + 1}: No JSON found, retrying...")
                        continue
                    raise ValueError("LLM did not return valid JSON")
                
                # Get the JSON string
                json_str = json_match.group()
                
                try:
                    # Parse the JSON into Python objects
                    self.metrics['json_parse_attempts'] += 1
                    suggestions = json.loads(json_str, strict=False)
                    
                    # Validate structure
                    if self._validate_suggestions(suggestions):
                        self.metrics['features_suggested'] += len(suggestions)
                        self.logger.info(f"Successfully parsed {len(suggestions)} feature suggestions")
                        print(f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} Successfully parsed {len(suggestions)} feature suggestions")
                        return suggestions
                    else:
                        if attempt < max_retries - 1:
                            print(f"⚠ Attempt {attempt + 1}: Invalid structure, retrying...")
                            continue
                        raise ValueError("Suggestions have invalid structure")
                        
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        print(f"⚠ Attempt {attempt + 1}: JSON error - {e}")
                        print(f"   Asking LLM to fix the JSON...")
                        
                        # Ask LLM to repair the broken JSON
                        repaired_content = self._ask_llm_to_fix_json(json_str, str(e))
                        
                        # Try to extract and parse the repaired JSON
                        json_match_repaired = re.search(r'\[.*\]', repaired_content, re.DOTALL)
                        if json_match_repaired:
                            json_str = json_match_repaired.group()
                            try:
                                suggestions = json.loads(json_str, strict=False)
                                if self._validate_suggestions(suggestions):
                                    print(f"✓ LLM successfully repaired the JSON!")
                                    return suggestions
                            except:
                                pass  # Will retry on next iteration
                        
                        print(f"   Repair unsuccessful, retrying from scratch...")
                        continue
                    else:
                        # Last attempt failed
                        print(f"✗ All {max_retries} attempts failed")
                        print(f"Last error: {e}")
                        print(f"Problematic JSON (first 500 chars):\n{json_str[:500]}")
                        raise ValueError(
                            f"LLM returned malformed JSON after {max_retries} attempts. "
                            f"Last error: {e}. "
                            "Try using llama3.1:8b or a different model."
                        )
                    
            except Exception as e:
                if "model" in str(e).lower() and "not found" in str(e).lower():
                    self._unload_model()  # Clean up on error
                    raise ValueError(
                        f"Model '{self.model}' not found. "
                        f"Please pull it first: ollama pull {self.model}"
                    )
                if attempt == max_retries - 1:
                    self._unload_model()  # Clean up on final failure
                    raise  # Re-raise on last attempt
                print(f"⚠ Attempt {attempt + 1} failed: {e}")
                continue
        
        # Should not reach here, but just in case
        self._unload_model()  # Clean up on unexpected failure
        raise ValueError(f"Failed to get valid suggestions after {max_retries} attempts")
    
    def apply_features(self, df: pd.DataFrame, suggestions: List[Dict[str, str]], 
                       safe_mode: bool = True) -> pd.DataFrame:
        """
        Apply suggested feature engineering transformations to the dataset.
        
        This method executes each suggested feature engineering transformation
        safely with error handling. Failed transformations are reported but
        don't stop the process (unless safe_mode=False).
        
        Safety considerations:
        - Each feature code runs in isolation
        - Errors are caught and reported
        - Original dataframe is not modified (uses .copy())
        - Failed features don't prevent other features from being created
        
        Args:
            df (pd.DataFrame): Input dataset (will not be modified)
            suggestions (List[Dict[str, str]]): Output from get_feature_suggestions()
            safe_mode (bool, optional): If True, continue on errors. If False, raise
                                       on first error. Default: True
        
        Returns:
            pd.DataFrame: New dataframe with original columns + engineered features
        
        Raises:
            Exception: Only if safe_mode=False and an error occurs
        
        Example:
            >>> df_original = pd.read_csv("data.csv")
            >>> suggestions = engineer.get_feature_suggestions(df_original)
            >>> df_enhanced = engineer.apply_features(df_original, suggestions)
            >>> print(f"Original shape: {df_original.shape}")
            >>> print(f"Enhanced shape: {df_enhanced.shape}")
        """
        
        # Create a copy to avoid modifying the original dataframe
        df_new = df.copy()
        successful_features = []  # Track features that were created successfully
        
        self.logger.info(f"Applying {len(suggestions)} feature suggestions")
        
        # Apply each suggested feature one by one
        for suggestion in suggestions:
            feature_name = suggestion['feature_name']
            code = suggestion['code']
            explanation = suggestion['explanation']
            
            try:
                # Execute the feature engineering code
                # The exec() function runs the code in a controlled namespace
                # We pass:
                # - 'df': the current dataframe (allows df['new_col'] = ... syntax)
                # - 'pd': pandas library (for operations like pd.cut, pd.factorize)
                # - 'np': numpy library (for functions like np.log, np.sqrt)
                exec(code, {'df': df_new, 'pd': pd, 'np': np})
                
                # Record this successful feature
                successful_features.append({
                    'name': feature_name,
                    'explanation': explanation,
                    'code': code
                })
                self.metrics['features_created'] += 1
                self.logger.info(f"Created feature: {feature_name}")
                print(f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} Created feature: {feature_name}")
                
            except Exception as e:
                # Handle errors gracefully
                self.metrics['features_failed'] += 1
                error_msg = f"{feature_name}: {str(e)}"
                self.metrics['errors'].append(error_msg)
                self.logger.warning(f"Failed to create {feature_name}: {str(e)}")
                print(f"{Colors.BRIGHT_RED}✗{Colors.RESET} Failed to create {feature_name}: {str(e)}")
                # Re-raise if not in safe mode
                if not safe_mode:
                    raise
        
        # Store the successful features for later reference
        # Note: This overwrites previous features, which is intentional for single-pass usage
        # For iterative usage, see iterative_feature_engineering which tracks all features
        self.generated_features = successful_features
        self.logger.info(f"Applied {len(successful_features)}/{len(suggestions)} features successfully")
        return df_new
    
    def iterative_feature_engineering(self, df: pd.DataFrame, target: str,
                                     task_type: str = "classification",
                                     iterations: int = 2) -> pd.DataFrame:
        """
        Perform multiple rounds of feature engineering iteratively.
        
        This advanced method performs feature engineering in multiple passes:
        - Iteration 1: Generates features from original data
        - Iteration 2: Generates features from original + iteration 1 features
        - And so on...
        
        This approach allows for:
        - Creating compound features (features of features)
        - Progressive data enrichment
        - More sophisticated feature interactions
        
        Example workflow:
        1. Original data: age, income
        2. Iteration 1: creates age_group, income_category
        3. Iteration 2: creates features using age_group + income_category
           (e.g., interaction between social class and age)
        
        Args:
            df (pd.DataFrame): Input dataset
            target (str): Target variable name for supervised learning context
            task_type (str, optional): "classification" or "regression". Default: "classification"
            iterations (int, optional): Number of feature engineering rounds. Default: 2
                                       More iterations = more features but longer runtime
        
        Returns:
            pd.DataFrame: Dataset with original columns + all engineered features
                         from all iterations
        
        Note:
            - Each iteration takes time (local LLM inference)
            - Feature explosion: n iterations can create 5-10 * n features
            - Use iterations=1 for quick results, iterations=3+ for thorough engineering
            - Local processing may be slower than cloud APIs but has no cost
        
        Example:
            >>> df = pd.read_csv("data.csv")
            >>> engineer = OllamaFeatureEngineer(model="llama3.2:latest")
            >>> df_enhanced = engineer.iterative_feature_engineering(
            ...     df,
            ...     target="sale_price",
            ...     task_type="regression",
            ...     iterations=2
            ... )
        """
        
        df_current = df.copy()  # Work with a copy
        all_features = []  # Track all features from all iterations
        
        self.logger.info(f"Starting iterative feature engineering: {iterations} iterations")
        
        try:
            # Run feature engineering for the specified number of iterations
            for i in range(iterations):
                print(f"\n=== Iteration {i+1}/{iterations} ===")
                self.logger.info(f"Starting iteration {i+1}/{iterations}")
                
                # Get suggestions from LLM based on current data state
                suggestions = self.get_feature_suggestions(df_current, target, task_type)
                
                # Apply the suggestions to create new features
                df_current = self.apply_features(df_current, suggestions)
                
                # Store all features created in this iteration
                all_features.extend(self.generated_features)
                self.metrics['iterations_completed'] += 1
            
            # Store ALL features from all iterations for explain_features()
            self.generated_features = all_features
            
            # Print summary statistics
            print(f"\n=== Summary ===")
            print(f"Total features created: {len(all_features)}")
            print(f"Final dataset shape: {df_current.shape}")
            print(f"  Original columns: {df.shape[1]}")
            print(f"  New columns: {df_current.shape[1] - df.shape[1]}")
            
            self.logger.info(f"Iterative feature engineering complete: {len(all_features)} features created")
            
            return df_current
            
        except Exception as e:
            # Unload model on any error during iterative engineering
            self.logger.error(f"Error during iterative feature engineering: {e}")
            self._unload_model()
            raise
    
    def print_metrics_summary(self) -> None:
        """
        Print a comprehensive summary of metrics collected during the session.
        
        Displays:
        - Session duration
        - LLM call statistics (count, total time, average time)
        - JSON parsing statistics
        - Feature engineering statistics
        - Success/failure rates
        
        Returns:
            None (prints to stdout)
        
        Example:
            >>> engineer.iterative_feature_engineering(df, target="price")
            >>> engineer.print_metrics_summary()
        """
        session_duration = (datetime.now() - self.metrics['session_start']).total_seconds()
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING METRICS SUMMARY")
        print("="*60)
        
        # Session info
        print(f"\nSession Duration: {session_duration:.2f}s ({session_duration/60:.2f} minutes)")
        print(f"Session Start: {self.metrics['session_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # LLM Statistics
        print(f"\nLLM Statistics:")
        print(f"  Total API Calls: {self.metrics['llm_calls']}")
        print(f"  Total LLM Time: {self.metrics['total_llm_time']:.2f}s")
        if self.metrics['llm_calls'] > 0:
            avg_time = self.metrics['total_llm_time'] / self.metrics['llm_calls']
            print(f"  Average Call Time: {avg_time:.2f}s")
            if self.metrics['llm_call_times']:
                min_time = min(self.metrics['llm_call_times'])
                max_time = max(self.metrics['llm_call_times'])
                print(f"  Fastest Call: {min_time:.2f}s")
                print(f"  Slowest Call: {max_time:.2f}s")
        
        # JSON Parsing Statistics
        print(f"\nJSON Parsing Statistics:")
        print(f"  Parse Attempts: {self.metrics['json_parse_attempts']}")
        print(f"  Parse Failures: {self.metrics['json_parse_failures']}")
        if self.metrics['json_parse_attempts'] > 0:
            success_rate = (1 - self.metrics['json_parse_failures'] / self.metrics['json_parse_attempts']) * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Repair Attempts: {self.metrics['json_repairs_attempted']}")
        print(f"  Successful Repairs: {self.metrics['json_repairs_successful']}")
        
        # Feature Engineering Statistics
        print(f"\nFeature Engineering Statistics:")
        print(f"  Features Suggested by LLM: {self.metrics['features_suggested']}")
        print(f"  Features Successfully Created: {self.metrics['features_created']}")
        print(f"  Features Failed: {self.metrics['features_failed']}")
        if self.metrics['features_suggested'] > 0:
            creation_rate = (self.metrics['features_created'] / self.metrics['features_suggested']) * 100
            print(f"  Creation Success Rate: {creation_rate:.1f}%")
        print(f"  Iterations Completed: {self.metrics['iterations_completed']}")
        
        # Error Summary
        if self.metrics['errors']:
            print(f"\nErrors Encountered: {len(self.metrics['errors'])}")
            for i, error in enumerate(self.metrics['errors'][:5], 1):  # Show first 5
                print(f"  {i}. {error}")
            if len(self.metrics['errors']) > 5:
                print(f"  ... and {len(self.metrics['errors']) - 5} more")
        else:
            print(f"\nErrors Encountered: 0")
        
        print("="*60 + "\n")
        
        # Log the summary as well
        self.logger.info("Metrics summary printed")
    
    def explain_features(self) -> None:
        """
        Print human-readable explanations of all generated features.
        
        This method displays the features created in the current session,
        including their names, the code that created them, and explanations
        of why they're useful. Useful for model interpretability and debugging.
        
        Output format:
        ```
        === Generated Features ===
        
        1. feature_name
           Code: df['new_col'] = df['col1'] * df['col2']
           Why: This captures interaction between col1 and col2
        
        2. another_feature
           Code: df['log_col'] = np.log1p(df['values'])
           Why: Log transformation handles skewed distributions
        ```
        
        Returns:
            None (prints to stdout)
        
        Example:
            >>> engineer.apply_features(df, suggestions)
            >>> engineer.explain_features()
        """
        # Check if any features were created
        if not self.generated_features:
            print("No features have been generated yet.")
            return
        
        # Print header
        print("\n=== Generated Features ===")
        
        # Print each feature with its explanation
        for i, feat in enumerate(self.generated_features, 1):
            print(f"\n{i}. {feat['name']}")
            print(f"   Code: {feat['code']}")
            print(f"   Why: {feat['explanation']}")


# ============================================================================
# EXAMPLE USAGE AND INTEGRATION WITH ML PIPELINE
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of how to use the OllamaFeatureEngineer in a complete
    machine learning workflow.
    
    PREREQUISITES:
    1. Install Ollama: https://ollama.ai/download
    2. Start Ollama service: ollama serve
    3. Pull a model: ollama pull llama3.2
    4. Install Python package: pip install ollama
    
    USAGE:
    # Basic usage with defaults
    python loaders/OllamaFeatureEngineer.py
    
    # Custom dataset and target
    python loaders/OllamaFeatureEngineer.py --dataset data/housing.csv --target price
    
    # Specify task type
    python loaders/OllamaFeatureEngineer.py --task-type regression
    
    # Multiple iterations with different model
    python loaders/OllamaFeatureEngineer.py --iterations 3 --model llama3.1:8b
    
    # Full example with all options
    python loaders/OllamaFeatureEngineer.py \
      --dataset data/custom.csv \
      --target sale_price \
      --task-type regression \
      --iterations 3 \
      --model mistral:latest
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform automated feature engineering using local Ollama LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/insurance.csv',
        help='Path to the CSV dataset file (default: data/insurance.csv)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='charges',
        help='Name of the target column (default: charges)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=2,
        help='Number of feature engineering iterations (default: 2)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:8b',
        help='Ollama model to use (default: llama3.1:8b)'
    )
    parser.add_argument(
        '--task-type',
        type=str,
        default=None,
        choices=['classification', 'regression', None],
        help='Type of ML task: "classification" or "regression" (default: auto-detect from target column)'
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    print(f"\n{Colors.BRIGHT_CYAN}Loading dataset: {args.dataset}{Colors.RESET}")
    try:
        df = pd.read_csv(args.dataset)
        print(f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print(f"{Colors.BRIGHT_RED}✗{Colors.RESET} Error: Dataset file not found: {args.dataset}")
        print(f"Please provide a valid path with --dataset flag")
        exit(1)
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}✗{Colors.RESET} Error loading dataset: {e}")
        exit(1)
    
    # Initialize the Ollama feature engineer
    print(f"\n{Colors.BRIGHT_CYAN}Initializing with model: {args.model}{Colors.RESET}")
    engineer = OllamaFeatureEngineer(
        model=args.model, 
        temperature=0.2  # Lower = more deterministic, better for JSON
    )
    
    # ========================================================================
    # OPTION 1: Single-pass feature engineering (quick)
    # ========================================================================
    # Uncomment to use single-pass instead of iterative
    """
    suggestions = engineer.get_feature_suggestions(
        df, 
        target="charges",  # Use actual column from insurance.csv
        task_type="regression"  # Predicting charges (continuous value)
    )
    
    # Apply the suggested features to the dataset
    df_enhanced = engineer.apply_features(df, suggestions)
    """
    
    # ========================================================================
    # OPTION 2: Iterative feature engineering (thorough - DEFAULT)
    # ========================================================================
    # Perform multiple rounds for more sophisticated features
    # Note: This will take longer with local models
    ## When to Use Each Type

    # Regression (`task_type="regression"`):
    # - Target is continuous/numeric
    # - Examples: price, temperature, charges, salary
    # - Goal: Predict a numeric value
    # Classification (`task_type="classification"`):

    # - Target is categorical/discrete
    # - Examples: yes/no, species, category labels
    # - Goal: Predict a class/category

    df_enhanced = engineer.iterative_feature_engineering(
        df,
        target=args.target,  # From --target flag
        task_type=args.task_type,  # From --task-type flag (None = auto-detect)
        iterations=args.iterations  # From --iterations flag
    )
    
    # ========================================================================
    # INSPECT CREATED FEATURES
    # ========================================================================
    # Show what features were created and why
    engineer.explain_features()
    
    # ========================================================================
    # VIEW METRICS SUMMARY
    # ========================================================================
    # Print comprehensive metrics about the session
    engineer.print_metrics_summary()
    
    print(f"\n✓ Feature engineering complete!")
    print(f"Original shape: {df.shape}")
    print(f"Enhanced shape: {df_enhanced.shape}")
    print(f"New features added: {df_enhanced.shape[1] - df.shape[1]}")
    print(f"New features: {df_enhanced.columns.tolist()[-(df_enhanced.shape[1] - df.shape[1]):]}")
