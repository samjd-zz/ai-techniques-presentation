# Machine Learning Data Loaders

This repository contains two comprehensive data loader classes for machine learning tasks: Decision Tree classification and K-Nearest Neighbors (KNN) with normalization.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Loaders](#data-loaders)
  - [LoadDecisionTree](#loaddecisiontree)
  - [LoadKNNNormalization](#loadknnnormalization)
- [LLM Feature Engineering](#llm-feature-engineering)
  - [OllamaFeatureEngineer](#ollamafeatureengineer)
- [Modular Architecture](#modular-architecture)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Generated Outputs](#generated-outputs)

---

## Overview

This project provides two specialized data loader modules designed for educational and practical machine learning applications:

1. **LoadDecisionTree** (`loaders/classification/LoadDecisionTree.py`): Decision Tree classification for insurance cost prediction
2. **LoadKNNNormalization** (`loaders/regression/LoadKNNNormalization.py`): K-Nearest Neighbors data preprocessing with normalization for smartphone activity recognition

Both loaders handle feature engineering, data preprocessing, visualization, and model training/evaluation.

---

## Installation

### Option 1: Using pip (requirements.txt)

```bash
# Clone or download the repository
git clone <repository-url>
cd ai-techniques-presentation

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda (environment.yml)

```bash
# Clone or download the repository
git clone <repository-url>
cd real-machine-learning

# Create and activate conda environment
conda env create -f environment.yml
conda activate ml-loaders
```

### Option 3: Manual Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Dependencies

- **Python 3.8+** (Python 3.11 recommended for conda environment)
- **NumPy ≥1.24.0**: Numerical computing
- **Pandas ≥2.0.0**: Data manipulation
- **Matplotlib ≥3.7.0**: Visualization
- **Seaborn ≥0.12.0**: Statistical data visualization
- **scikit-learn ≥1.3.0**: Machine learning algorithms

---

## Data Loaders

### LoadDecisionTree

**Purpose**: Classification of insurance charges into risk tiers using Decision Tree algorithms.

**Key Features**:

- Loads insurance CSV data (age, sex, bmi, children, smoker, region, charges)
- Automatic feature engineering:
  - Age groups (young_adult, middle_aged, senior_adult)
  - BMI groups (WHO standard categories)
  - Charge bins (10 equal-width categories)
- Regional data filtering (northeast, southeast, northwest, southwest)
- Decision Tree model training and evaluation
- Two rule extraction methods (custom with confidence scores, simple sklearn export)
- Tree visualization with feature importance
- Comprehensive performance metrics (confusion matrix, classification report)

**Why charges_group as Target?**

- **Business Value**: Risk stratification for premium setting and financial planning
- **Feature Alignment**: Strong predictors (age, BMI, smoker status) naturally correlate with costs
- **Ideal for Decision Trees**: 10 meaningful classes enable complex pattern detection
- **Real-World Applicability**: Direct use for applicant classification and budget forecasting

**Train-Test Split**:

```
Insurance Data (1338 rows)
    ↓
Split into:
├─ Training Set (70% = ~936 rows)  ← Train models
└─ Testing Set (30% = ~402 rows)   ← Evaluate models
```

The data is split, then 12 different Decision Tree configurations are trained on the training set and evaluated on the test set to find the best model.

---

### LoadKNNNormalization

**Purpose**: Preprocessing and normalization for K-Nearest Neighbors classification of human activity using smartphone sensor data.

**Key Features**:

- Loads smartphone sensor CSV data (561 features from accelerometer/gyroscope)
- Subject grouping (groups of 5 subjects: 1-5, 6-10, ..., 26-30)
- Activity-based filtering (WALKING, SITTING, STANDING, etc.)
- Z-score normalization (standardization: mean=0, std=1)
- Statistical analysis:
  - Activity statistics (mean, std, min, max)
  - Subject statistics
  - Correlation matrices
- **Outlier detection and removal** using z-score method
- Euclidean distance computation (for KNN)
- Balanced sampling (equal samples per activity class)
- Train/test splitting with stratification
- Visualization tools (activity features, scatter plots)

**Normalization**:

```python
z = (x - μ) / σ
# Where:
# x = original value
# μ = mean
# σ = standard deviation
```

---

## LLM Feature Engineering

### OllamaFeatureEngineer

**Purpose**: Privacy-preserving automated feature engineering using locally-running LLMs via Ollama - no API costs, data never leaves your machine.

**Location**: `loaders/OllamaFeatureEngineer.py`

**Key Features**:

- **100% Local**: Runs entirely on your machine via Ollama
- **No API Costs**: Free after initial setup (uses local compute)
- **Privacy-Preserving**: Your data never leaves your computer
- **Same Interface**: Drop-in replacement for LLMFeatureEngineer
- **Multiple Models**: Support for Llama, Mistral, Qwen, DeepSeek, and more
- **🎨 Colorful Logging**: ANSI color-coded output with Unicode icons for visual clarity
- **📊 Comprehensive Metrics**: Track LLM calls, timing, success rates, and errors
- **🧹 Automatic Cleanup**: Model unloading on errors to free memory
- **🎯 Smart Detection**: Auto-detects task type from target column
- **⚙️ CLI Support**: Full command-line interface for all parameters

**Prerequisites**:
```bash
# 1. Install Ollama (https://ollama.ai/download)
# For Linux:
curl -fsSL https://ollama.com/install.sh | sh

# For macOS:
brew install ollama

# For Windows: Download from https://ollama.ai/download

# 2. Start Ollama service
ollama serve

# 3. Pull a model (in another terminal)
ollama pull llama3.2  # 3B model, fast and capable

# 4. Install Python package
pip install ollama>=0.4.0
# Or: conda run -n ml-loaders pip install ollama>=0.4.0
```

**Quick Example**:
```python
from loaders.OllamaFeatureEngineer import OllamaFeatureEngineer
import pandas as pd

# Load your dataset
df = pd.read_csv("data/insurance.csv")

# Initialize (no API key needed!)
engineer = OllamaFeatureEngineer(
    model="llama3.2:latest",  # Fast 3B model
    temperature=0.3  # Lower = more deterministic
)

# Get AI-powered feature suggestions
suggestions = engineer.get_feature_suggestions(
    df,
    target="charges",
    task_type="regression"
)

# Apply the suggested features
df_enhanced = engineer.apply_features(df, suggestions)

# See what was created
engineer.explain_features()

print(f"Original: {df.shape[1]} features")
print(f"Enhanced: {df_enhanced.shape[1]} features")
```

**Supported Models** (must install with `ollama pull <model>`):
- `llama3.2:latest` (3B): Fast, good quality - **DEFAULT**
- `llama3.1:latest` (8B): Better quality, slower
- `mistral:latest` (7B): Balanced performance
- `qwen2.5-coder:latest` (7B): Optimized for code generation
- `deepseek-coder-v2:latest` (16B): Best quality, needs 16GB+ RAM

**Memory Requirements**:
- 3B models: ~4GB RAM
- 7B models: ~8GB RAM
- 13B+ models: ~16GB RAM

**Performance Tips**:
- First run may be slow (model loading)
- Subsequent runs are faster (model stays in memory)
- Use smaller models (llama3.2) for faster iteration
- Use larger models (deepseek-coder-v2) for better quality

**Command-Line Interface**:

OllamaFeatureEngineer now supports full CLI configuration:

```bash
# Basic usage with defaults
python loaders/OllamaFeatureEngineer.py

# Custom dataset and target
python loaders/OllamaFeatureEngineer.py --dataset data/housing.csv --target price

# Specify task type (classification or regression)
python loaders/OllamaFeatureEngineer.py --task-type regression

# Multiple iterations with different model
python loaders/OllamaFeatureEngineer.py --iterations 3 --model llama3.1:8b

# Full example
python loaders/OllamaFeatureEngineer.py \
  --dataset data/custom.csv \
  --target sale_price \
  --task-type regression \
  --iterations 3 \
  --model mistral:latest
```

**CLI Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/insurance.csv` | Path to CSV dataset |
| `--target` | `charges` | Target column name |
| `--task-type` | `None` (auto-detect) | `classification` or `regression` |
| `--iterations` | `2` | Number of feature engineering rounds |
| `--model` | `llama3.1:8b` | Ollama model to use |

**Enhanced Features**:

1. **🎨 Colorful Logging**:
   - 🟢 Green INFO messages with ℹ icon
   - 🟡 Yellow WARNING messages with ⚠ icon
   - 🔴 Red ERROR messages with ✗ icon
   - Green ✓ for successful operations
   - Red ✗ for failed operations

2. **📊 Comprehensive Metrics**:
   ```
   FEATURE ENGINEERING METRICS SUMMARY
   ====================================
   Session Duration: 595.04s (9.92 minutes)
   
   LLM Statistics:
     Total API Calls: 3
     Average Call Time: 198.33s
     Fastest Call: 156.09s
     Slowest Call: 270.07s
   
   Feature Engineering Statistics:
     Features Suggested by LLM: 29
     Features Successfully Created: 23
     Creation Success Rate: 79.3%
   ```

3. **🧹 Automatic Cleanup**:
   - Model automatically unloaded on errors
   - Frees memory to prevent resource leaks
   - Graceful error handling

4. **🎯 Smart Task Detection**:
   - Auto-detects regression vs classification
   - Based on target column data type
   - Numeric with >10 unique values → regression
   - Categorical or ≤10 values → classification

---

### Iterative Feature Engineering

OllamaFeatureEngineer supports iterative feature engineering for more sophisticated transformations:

```python
df_enhanced = engineer.iterative_feature_engineering(
    df,
    target="charges",
    task_type="regression",
    iterations=2  # Multiple rounds of feature creation
)

# Iteration 1: Creates features from original data
#   Example: age_squared, bmi_category, smoker_charges_interaction
#
# Iteration 2: Creates features from original + iteration 1 features
#   Example: age_squared_log, bmi_category_smoker_interaction
```

**Benefits of Iterative Engineering**:
- Creates compound features (features of features)
- Discovers complex interactions
- Progressively enriches the dataset

**Trade-offs**:
- More features = more training time
- More iterations = more compute time
- Risk of overfitting with too many features

---

## Modular Architecture

The Decision Tree loader has been refactored into a modular architecture for better maintainability and reusability:

### Core Modules

1. **LoadDecisionTree** (`loaders/classification/LoadDecisionTree.py`)
   - Main interface for users
   - Handles data loading and regional filtering
   - Coordinates feature engineering, aggregation, and modeling
   - ~750 lines (reduced from ~1100 lines)

2. **FeatureEngineer** (`loaders/classification/FeatureEngineer.py`)
   - Responsible for all feature engineering tasks
   - Creates age groups, BMI groups, charge bins, and index
   - Standalone utility that can be used independently
   - ~170 lines

3. **DataAggregator** (`loaders/classification/DataAggregator.py`)
   - Handles data aggregation and regional comparisons
   - Generates bar charts and statistical summaries
   - Can work with any regional dataset structure
   - ~170 lines

4. **TreeVisualizer** (`loaders/classification/TreeVisualizer.py`)
   - Dedicated to decision tree visualization
   - Two visualization styles: sklearn-style and RapidMiner-style
   - Can visualize any trained Decision Tree model
   - ~600 lines

### Benefits of Modular Design

✅ **Separation of Concerns** - Each module has a single, clear responsibility  
✅ **Reusability** - Modules can be used independently in other projects  
✅ **Maintainability** - Smaller, focused files are easier to understand and modify  
✅ **Testability** - Each module can be tested in isolation  
✅ **No Code Duplication** - Single source of truth for each function  

### Usage with New Architecture

```python
from loaders.classification.LoadDecisionTree import LoadDecisionTree
from loaders.classification.TreeVisualizer import DecisionTreeVisualizer

# Main loader handles everything through delegation
dt_loader = LoadDecisionTree()  # Uses default path
dt_loader.modeling_and_evaluation()

# Visualizer is now a separate class
visualizer = DecisionTreeVisualizer(
    model=dt_loader.best_model,
    X_train=dt_loader.X_train,
    label_encoders=dt_loader.label_encoders,
    target_encoder=dt_loader.target_encoder
)

# Two visualization methods available
visualizer.visualize_sklearn_style(save_path='plots/tree.png')
visualizer.visualize_rapidminer_style(save_path='plots/tree_decoded.png')
```

---

## Quick Start

### Decision Tree Classification (Insurance Data)

```python
from loaders.classification.LoadDecisionTree import LoadDecisionTree

# Initialize loader
dt_loader = LoadDecisionTree()

# Get regional data
southeast = dt_loader.get_southeast_df()
print(f"Southeast records: {len(southeast)}")

# Get aggregated statistics
charges_by_smoker = dt_loader.get_aggregate_tbl(grp_by='smoker', agg_by='charges')
print(charges_by_smoker)

# Generate visualization
dt_loader.get_grpd_bar_chart(grp_by='children', agg_by='charges')

# Run full modeling pipeline
dt_loader.modeling_and_evaluation()

# Extract decision rules (two methods)
dt_loader.extract_rules(max_rules=10)           # Custom with confidence
dt_loader.extract_rules_simple(max_depth=4)     # Simple tree structure

# Visualize tree (using TreeVisualizer)
from loaders.classification.TreeVisualizer import DecisionTreeVisualizer

visualizer = DecisionTreeVisualizer(
    model=dt_loader.best_model,
    X_train=dt_loader.X_train,
    label_encoders=dt_loader.label_encoders,
    target_encoder=dt_loader.target_encoder
)
visualizer.visualize_sklearn_style('plots/tree.png')
visualizer.visualize_rapidminer_style('plots/tree_decoded.png')
```

### KNN with Normalization (Smartphone Activity Data)

```python
from loaders.regression.LoadKNNNormalization import LoadKNNNormalization

# Initialize loader
knn_loader = LoadKNNNormalization('data/human-smartphones.csv')

# Get activity statistics
walking_stats = knn_loader.compute_activity_statistics('WALKING')
print("Walking - Mean of first 5 features:", walking_stats['mean'][:5])

# Detect and remove outliers BEFORE normalization
outliers = knn_loader.find_outliers_zscore(threshold=3)
print(f"Outliers detected: {len(outliers)}")

_, removed_count = knn_loader.remove_outliers(threshold=3, inplace=True)
print(f"Outliers removed: {removed_count}")

# Normalize features AFTER outlier removal (z-score standardization)
normalized = knn_loader.normalize_features()
print(f"Normalized shape: {normalized.shape}")

# Create balanced dataset
balanced = knn_loader.sample_balanced(n_per_activity=100, random_seed=42)
print(f"Balanced sample: {balanced.shape}")

# Split data
train, test = knn_loader.split_train_test(train_ratio=0.8, random_seed=42)
print(f"Train: {train.shape}, Test: {test.shape}")

# Generate visualizations
knn_loader.plotActivity()              # Activity feature means
knn_loader.plotFeaturesScatter()       # Feature scatter by activity
```

---

## Detailed Usage

### LoadDecisionTree Methods

#### Data Access Methods

```python
# Get regional dataframes
southeast_df = dt_loader.get_southeast_df()
northeast_df = dt_loader.get_northeast_df()
southwest_df = dt_loader.get_southwest_df()
northwest_df = dt_loader.get_northwest_df()

# Get aggregated tables
agg_table = dt_loader.get_aggregate_tbl(
    grp_by='children',    # Group by: 'children', 'smoker', 'age_group', 'bmi_group'
    agg_by='charges'      # Aggregate: 'charges'
)
```

#### Visualization Methods

```python
# Generate grouped bar chart
plot = dt_loader.get_grpd_bar_chart(
    grp_by='smoker',      # X-axis grouping
    agg_by='charges'      # Y-axis values
)
```

#### Modeling Methods

```python
# Train and evaluate 12 Decision Tree models
dt_loader.modeling_and_evaluation()

# Extract rules - Custom method (with confidence scores)
rules = dt_loader.extract_rules(max_rules=10)

# Extract rules - Simple sklearn method (tree structure)
tree_text = dt_loader.extract_rules_simple(max_depth=4)

# Visualize decision tree
dt_loader.visualize_best_tree(
    save_path='plots/decision_tree.png',
    max_depth=3  # Limit depth for readability
)
```

#### What Happens During modeling_and_evaluation()

1. **Prepare Data** (automatic):
   
   - Select features based on target (default: charges_group)
   - Encode categorical variables using Label Encoding
   - Split into train (70%) and test (30%) with stratification

2. **Train Models**:
   
   - Test 12 different hyperparameter combinations
   - Each model trains on training set
   - Evaluate on test set

3. **Evaluate Best Model**:
   
   - Display confusion matrix
   - Show classification report (precision, recall, F1-score)
   - Report accuracy and hyperparameters

---

### LoadKNNNormalization Methods

#### Data Access Methods

```python
# Get subject groups (1-6)
group1 = knn_loader.get_subject_group(1)  # Subjects 1-5
group2 = knn_loader.get_subject_group(2)  # Subjects 6-10

# Get features by activity
walking_features = knn_loader.get_features_by_activity('WALKING')
sitting_features = knn_loader.get_features_by_activity('SITTING')
```

#### Statistical Methods

```python
# Compute activity statistics
stats = knn_loader.compute_activity_statistics('WALKING')
# Returns: {'mean': array, 'std': array, 'min': array, 'max': array}

# Get subject statistics
subj_stats = knn_loader.get_subject_statistics(subject_id=5)
# Returns: {'count': int, 'mean': array, 'std': array}

# Correlation matrix
corr_matrix = knn_loader.correlation_matrix(feature_indices=[0, 1, 2])
```

#### Preprocessing Methods

```python
# Normalize features (z-score: mean=0, std=1)
normalized = knn_loader.normalize_features()

# Find outliers (z-score > threshold)
outlier_indices = knn_loader.find_outliers_zscore(threshold=3)

# Remove outliers from the dataset
_, removed_count = knn_loader.remove_outliers(threshold=3, inplace=True)
print(f"Removed {removed_count} outliers")

# Filter by threshold
filtered = knn_loader.filter_by_threshold(
    column_name='tBodyAcc-mean()-X',
    threshold=0.5,
    operator='greater'  # 'greater', 'less', 'equal'
)
```

#### Sampling and Splitting Methods

```python
# Create balanced dataset
balanced = knn_loader.sample_balanced(
    n_per_activity=100,
    random_seed=42
)

# Split train/test
train, test = knn_loader.split_train_test(
    train_ratio=0.8,
    random_seed=42
)
```

#### Distance Methods (for KNN)

```python
# Compute Euclidean distances from sample to all others
distances = knn_loader.compute_euclidean_distances(sample_idx=0)
```

#### Visualization Methods

```python
# Plot mean feature values by activity
knn_loader.plotActivity()  # Saves to plots/mean.png

# Scatter plot of first two features
knn_loader.plotFeaturesScatter()  # Saves to plots/first_two_features.png
```

---

## Generated Outputs

### Directory Structure

```
.
├── data/
│   ├── insurance.csv            # Insurance dataset
│   └── human-smartphones.csv    # Smartphone activity dataset
├── plots/
│   ├── children_charges.png     # Insurance charges by children
│   ├── smoker_charges.png       # Insurance charges by smoker status
│   ├── age_group_charges.png    # Insurance charges by age group
│   ├── bmi_group_charges.png    # Insurance charges by BMI group
│   ├── decision_tree.png        # Decision tree visualization
│   ├── mean.png                 # Activity feature means
│   └── first_two_features.png   # Feature scatter plot
├── loaders/
│   ├── classification/
│   │   ├── LoadDecisionTree.py      # Decision Tree loader (main interface)
│   │   ├── FeatureEngineer.py       # Feature engineering utilities
│   │   ├── DataAggregator.py        # Data aggregation and visualization
│   │   └── TreeVisualizer.py        # Decision tree visualization
│   └── regression/
│       └── LoadKNNNormalization.py  # KNN normalization loader
└── README.md                        # This file
```

### Plot Examples

#### Decision Tree Outputs

- **Regional Comparison Charts**: Bar charts showing charges by various factors across regions
- **Decision Tree Visualization**: Tree diagram showing split points and decision rules
- **Feature Importance**: Bar chart ranking features by importance

#### KNN Outputs

- **Activity Feature Means**: Line plot showing mean sensor values per activity
- **Feature Scatter**: Scatter plot of first two features colored by activity type

---

## Important Concepts

### Label Encoding vs One-Hot Encoding

**Label Encoding** (used in LoadDecisionTree):

- Converts categories to integers: `['male', 'female'] → [1, 0]`
- ✅ **Works for Decision Trees**: Trees treat numbers as categories
- ❌ **Don't use for**: Linear Regression, Neural Networks (implies false ordering)

**One-Hot Encoding** (alternative):

- Creates binary columns: `sex_male=[1,0], sex_female=[0,1]`
- ✅ **Use for**: Linear models, Neural Networks, Distance-based algorithms
- ❌ **Drawback**: More memory, more features

### Z-Score Normalization

Used in LoadKNNNormalization:

```
z = (x - μ) / σ

Where:
- x = original value
- μ = mean of feature
- σ = standard deviation of feature
```

**Purpose**:

- Standardizes features to mean=0, std=1
- Essential for distance-based algorithms (KNN, SVM)
- Prevents features with large scales from dominating

### Train-Test Split

**Why split?**

- **Training Set** (70%): Learn patterns
- **Testing Set** (30%): Evaluate on unseen data
- **Prevents overfitting**: Tests generalization ability

**Stratification**: Maintains class distribution in both sets

---

## Tips and Best Practices

### For Decision Tree Classification

1. **Start with defaults**: Run `modeling_and_evaluation()` first
2. **Interpret rules**: Use both extraction methods to understand decisions
3. **Check feature importance**: Visualize tree to see key decision points
4. **Handle imbalance**: If needed, use class weights or resampling

### For KNN Normalization

1. **Always normalize**: KNN is distance-based, normalization is crucial
2. **Remove outliers**: Use `remove_outliers()` to clean data before training
3. **Balance classes**: Use `sample_balanced()` for imbalanced data
4. **Set random seed**: Ensure reproducibility in splits and sampling

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary

- ✅ You can use this for commercial purposes
- ✅ You can modify and distribute the code
- ✅ You must include the license and copyright notice
- ❌ The software is provided "as is" with no warranty

## Contact

For questions or issues, please refer to the inline documentation in each loader class.

---

**Happy Machine Learning! 🚀**
