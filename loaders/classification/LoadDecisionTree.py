#!/usr/bin/env python3

# %matplotlib inline  # Jupyter notebook magic command - commented out for standalone Python script
import matplotlib
matplotlib.use('Agg')  # Set backend FIRST
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree as sklearn_tree
from sklearn.tree import export_text

# Import refactored modules
from FeatureEngineer import FeatureEngineer
from DataAggregator import DataAggregator

class LoadDecisionTree:
    """
    A data loader and analyzer for Decision Tree classification tasks using insurance data.
    
    Handles loading insurance CSV data, creating age group features, filtering by region,
    and providing aggregated views and visualizations for classification analysis.
    """
    
    # constructor ------------------------------------------------------------#
    def __init__(self, csv_data='data/insurance.csv'):
        """
        Initialize the Decision Tree classification loader with insurance data.
        
        Args:
            csv_data (str): Path to CSV file containing insurance data with columns like
                           age, sex, bmi, children, smoker, region, charges
        """
        # Load data
        df = pd.read_csv(csv_data)
        
        # STEP 1: Add ALL features using FeatureEngineer
        engineer = FeatureEngineer()
        df = engineer.add_all_features(df)
        
        # STEP 2: split regions
        self.southeast_df = self._region_filter(df,'southeast')
        self.northeast_df = self._region_filter(df,'northeast')
        self.southwest_df = self._region_filter(df,'southwest')
        self.northwest_df = self._region_filter(df,'northwest')
        
        # STEP 3: Create DataAggregator for regional analysis
        regional_dfs = {
            'northeast': self.northeast_df,
            'southeast': self.southeast_df,
            'northwest': self.northwest_df,
            'southwest': self.southwest_df
        }
        self.aggregator = DataAggregator(regional_dfs)
        
        # Store full dataset for modeling
        self.full_df = df
        
        # Initialize modeling attributes
        self.label_encoders = {}
        self.best_accuracy = 0.0
        self.best_model = None
        self.results_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    # private methods ---------------------------------------------------------#
    def _region_filter(self, df, region):
        """
        Filter dataframe to include only rows from specified region.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'region' column
            region (str): Region name to filter ('northeast', 'southeast', 'northwest', 'southwest')
            
        Returns:
            pd.DataFrame: Filtered copy containing only records from specified region
        """
        return df[df['region'] == region].copy()

    def _prepare_for_modeling(self, target='charges_group', use_grouped_features_only=False):
        """
        Encode categorical variables and split data for modeling.
        
        This method prepares the insurance data for Decision Tree classification by:
        1. Selecting appropriate features based on the target variable
        2. Encoding all categorical variables into numeric values
        3. Splitting data into training (70%) and testing (30%) sets
        4. Displaying encoding mappings and class distributions
        
        Args:
            target (str): Target variable for classification (default: 'charges_group')
                         Options: 'charges_group', 'smoker', 'age_group', 'bmi_group'
            use_grouped_features_only (bool): If True, exclude numeric age/bmi and only use
                                              age_group/bmi_group. Default: False
        """
        print(f"\n" + "="*50)
        print("PREPARING DATA FOR MODELING")
        print(f"Target variable: {target}")
        if use_grouped_features_only:
            print("Mode: GROUPED FEATURES ONLY (no numeric age/bmi)")
        else:
            print("Mode: ALL FEATURES (numeric + grouped)")
        print("="*50)
        
        # Reset index to convert ID back to a column, making all features accessible
        # This is necessary because the index was set during initialization
        df = self.full_df.reset_index()
        
        # ============================================================
        # STEP 1: Feature Selection
        # ============================================================
        # Start with base features
        if use_grouped_features_only:
            # GROUPED MODE: Exclude numeric age and bmi
            feature_cols = ['sex', 'children', 'smoker', 'region']
            print("  → Excluding numeric: age, bmi")
        else:
            # FULL MODE: Include all numeric features
            feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        
        # Add engineered features ONLY if they are NOT the target variable
        # This prevents data leakage (using the target to predict itself)
        # Example: If predicting 'smoker', we can use 'age_group' and 'bmi_group' as features
        if target != 'age_group':
            feature_cols.append('age_group')
            if use_grouped_features_only:
                print("  → Including grouped: age_group")
        if target != 'bmi_group':
            feature_cols.append('bmi_group')
            if use_grouped_features_only:
                print("  → Including grouped: bmi_group")
        if target != 'charges_group':
            feature_cols.append('charges_group')
        
        # ============================================================
        # STEP 2: Separate Features (X) and Target (y)
        # ============================================================
        # X contains all predictor variables (features)
        # y contains the variable we want to predict (target)
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # ============================================================
        # STEP 3: Encode Categorical Features
        # ============================================================
        # Decision Trees in sklearn require numeric input, so we must convert
        # categorical variables (strings) to numeric codes using LABEL ENCODING
        #
        # NOMINAL-TO-NUMERIC ENCODING (Label Encoding):
        # - Converts categorical labels to integer codes
        # - Example: ['male', 'female'] -> [1, 0] OR ['northeast', 'southeast', 'northwest', 'southwest'] -> [0, 1, 2, 3]
        # 
        # WHY THIS WORKS FOR DECISION TREES:
        # - Decision trees treat encoded values as CATEGORIES, not mathematical relationships
        # - They don't assume that 1 > 0 or that northwest (3) > northeast (0)
        # - Trees split on categorical values, not numerical relationships
        #
        # IMPORTANT CAVEAT - When NOT to use Label Encoding:
        # - Linear Regression: Would treat 1 > 0 as meaningful math → Wrong results
        # - Neural Networks: Would learn numerical patterns that don't exist → Wrong results
        # - Distance-based models (KNN, SVM): Would use numeric distance → Wrong results
        # - For these algorithms, use ONE-HOT ENCODING instead:
        #   * Creates binary columns for each category (no implied ordering)
        #   * Example: sex_male=[1,0,0,...], sex_female=[0,1,0,...]
        #
        # WHY LABEL ENCODING IS FINE HERE:
        # - Decision trees handle it correctly (categorical, not ordinal treatment)
        # - More memory efficient than one-hot encoding
        # - Simpler to interpret and store encoders
        X_encoded = X.copy()
        self.label_encoders = {}  # Store encoders for later use (e.g., interpreting results)
        
        # Loop through each feature column
        for column in X.columns:
            # Check if the column contains categorical (string) data
            if X[column].dtype == 'object':
                # Create a new LabelEncoder for this column
                le = LabelEncoder()
                # Fit the encoder to the unique values and transform them to numeric codes
                # This is LABEL ENCODING: categorical → numeric (for Decision Tree compatibility)
                # Example: ['male', 'female'] -> [1, 0]
                X_encoded[column] = le.fit_transform(X[column])
                # Save the encoder so we can decode predictions later
                self.label_encoders[column] = le
                # Print the mapping for transparency
                # Example: "Encoded sex: {'female': 0, 'male': 1}"
                print(f"Encoded {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # ============================================================
        # STEP 4: Encode Target Variable
        # ============================================================
        # The target variable also needs to be encoded to numeric values using LABEL ENCODING
        # This is also NOMINAL-TO-NUMERIC ENCODING (same as features above)
        # 
        # The target variable (what we're predicting) must be numeric for Decision Trees
        # Example: 'charges_bin_1', 'charges_bin_2', ..., 'charges_bin_10', 'Free' -> [0, 1, ..., 9, 10]
        # 
        # NOTE: Decision trees accept the numeric codes without treating them as ordinal,
        # so this is appropriate for multiclass classification targets as well.
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        # Print the target class mapping
        # Example: "Target classes: {'free': 0, 'charges_bin_1': 1, ..., 'charges_bin_10': 10}"
        target_mapping = dict(zip(self.target_encoder.classes_, 
                                  self.target_encoder.transform(self.target_encoder.classes_)))
        print(f"\nTarget classes: {target_mapping}")
        
        # ============================================================
        # STEP 5: Train-Test Split
        # ============================================================
        # Split the data into:
        # - Training set (70%): Used to train the Decision Tree models
        # - Testing set (30%): Used to evaluate model performance on unseen data
        # 
        # Key parameters:
        # - test_size=0.3: Reserve 30% of data for testing
        # - random_state=42: Ensures reproducible splits across runs
        # - stratify=y_encoded: Maintains the same class distribution in both sets
        #                       (prevents imbalanced splits where one set has all of one class)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # ============================================================
        # STEP 6: Display Split Information
        # ============================================================
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
        # Show the distribution of classes in the training set
        # This helps verify that stratification worked correctly
        print(f"\nClass distribution in training set:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            # Convert numeric code back to original class name for readability
            class_name = self.target_encoder.inverse_transform([cls])[0]
            print(f"  {class_name}: {count} ({count/len(self.y_train):.1%})")
    
    # public methods ---------------------------------------------------------#
    
    # ============================================================
    # TARGET VARIABLE SELECTION: Why charges_group?
    # ============================================================
    # charges_group was selected as the default target variable for the following reasons:
    #
    # 1. BUSINESS VALUE: Insurance companies need to predict and stratify healthcare costs
    #    for premium setting, resource allocation, and financial planning. Classifying
    #    customers into 10 charge bins (low to high) enables risk stratification.
    #
    # 2. FEATURE ALIGNMENT: Strong predictive features naturally correlate with costs:
    #    - Age: Older individuals have higher medical expenses
    #    - BMI: Higher BMI correlates with increased health risks
    #    - Smoker Status: Significant predictor of healthcare utilization
    #    - Region: Geographic variations in healthcare costs
    #
    # 3. IDEAL FOR DECISION TREES: 
    #    - Multiple meaningful classes (10 bins) without being overly granular
    #    - Decision trees excel at capturing non-linear relationships and interactions
    #    - Example: "high BMI + smoker" produces different outcome than either alone
    #
    # 4. AVOIDS COMMON PITFALLS:
    #    - No data leakage: charges_group is derived but independent from raw charges
    #    - Richer information than binary targets (better than just "high/low")
    #    - More practical than continuous regression for business decisions
    #
    # 5. REAL-WORLD APPLICABILITY:
    #    - Can classify new applicants into risk tiers for pricing
    #    - Enables case management for high-risk customers
    #    - Facilitates budget forecasting with granular predictions
    #
    # Alternative targets (smoker, age_group, bmi_group) were less ideal due to:
    #    - Fewer classes (2-4) limiting model complexity
    #    - Lower business relevance for insurance operations
    #    - Less direct connection to financial decision-making
    # ============================================================
    
    def modeling_and_evaluation(self, use_grouped_features_only=False):
        """
        Test multiple decision tree configurations and find the best model.
        
        This method performs a comprehensive evaluation by:
        1. Testing 12 different Decision Tree configurations
        2. Training each model on the training set
        3. Evaluating each model on the test set
        4. Identifying the best performing model
        5. Providing detailed metrics for the best model
        
        Args:
            use_grouped_features_only (bool): If True, only use grouped features (age_group, bmi_group)
                                              and exclude numeric age/bmi. Default: False (use all features)
        
        Must call prepare_for_modeling() first to set up train/test data.
        
        Raises:
            ValueError: If prepare_for_modeling() hasn't been called yet
        """

        self._prepare_for_modeling(use_grouped_features_only=use_grouped_features_only)  # Ensure data is prepared before modeling

        # ============================================================
        # STEP 0: Validation Check
        # ============================================================
        # Ensure that the data has been prepared before attempting to train models
        if self.X_train is None or self.y_train is None:
            raise ValueError("Must call prepare_for_modeling() before modeling_and_evaluation()")
        
        print(f"\n" + "="*60)
        print("MODELING & EVALUATION")
        print("="*60)
        
        # ============================================================
        # STEP 1: Define Hyperparameter Combinations
        # ============================================================
        # Test various Decision Tree configurations to find the best model
        # Each configuration varies key hyperparameters:
        # 
        # - criterion: Splitting criterion ('gini' for Gini impurity, 'entropy' for information gain)
        # - max_depth: Maximum tree depth (controls model complexity)
        # - min_samples_split: Minimum samples required to split a node (prevents overfitting)
        # - min_samples_leaf: Minimum samples required in a leaf node (prevents overfitting)
        # - ccp_alpha: Pruning parameter (higher values = more aggressive pruning)
        param_combinations = [
            {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 2, 'ccp_alpha': 0.0},
            {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 2, 'ccp_alpha': 0.0},
            {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 4, 'min_samples_leaf': 2, 'ccp_alpha': 0.01},
            {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 5, 'min_samples_leaf': 1, 'ccp_alpha': 0.0},
            {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 3, 'ccp_alpha': 0.005},
            {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6, 'min_samples_leaf': 2, 'ccp_alpha': 0.002},
            {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'ccp_alpha': 0.01},
            {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 4, 'min_samples_leaf': 4, 'ccp_alpha': 0.0},
            {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 7, 'min_samples_leaf': 2, 'ccp_alpha': 0.0},
            {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 3, 'ccp_alpha': 0.02},
            {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 5, 'min_samples_leaf': 1, 'ccp_alpha': 0.005},
            {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'ccp_alpha': 0.001}
        ]
        
        # Initialize storage for results
        results = []
        self.best_accuracy = 0.0
        
        # ============================================================
        # STEP 2: Train and Evaluate Each Configuration
        # ============================================================
        print("\nTesting different parameter combinations...")
        print(f"{'#':<3} {'Criterion':<10} {'Max Depth':<10} {'Min Split':<10} {'Min Leaf':<10} {'Pruning':<10} {'Accuracy':<10}")
        print("-" * 70)
        
        # Loop through each parameter combination
        for i, params in enumerate(param_combinations, 1):
            # Create a new Decision Tree with the current parameters
            model = DecisionTreeClassifier(
                criterion=params['criterion'],          # Splitting criterion
                max_depth=params['max_depth'],          # Maximum tree depth
                min_samples_split=params['min_samples_split'],  # Min samples to split node
                min_samples_leaf=params['min_samples_leaf'],    # Min samples in leaf
                ccp_alpha=params['ccp_alpha'],          # Pruning parameter
                random_state=42                         # For reproducibility
            )
            
            # Train the model on the training data
            # The tree learns patterns by recursively splitting the data
            model.fit(self.X_train, self.y_train)
            
            # Make predictions on the test set (unseen data)
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy: proportion of correct predictions
            # Accuracy = (Correct Predictions) / (Total Predictions)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store the results for this configuration
            result = params.copy()
            result['accuracy'] = accuracy
            result['model'] = model  # Save the trained model
            results.append(result)
            
            # Track the best performing model so far
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
            
            # Print this configuration's results
            print(f"{i:<3} {params['criterion']:<10} {params['max_depth']:<10} {params['min_samples_split']:<10} "
                  f"{params['min_samples_leaf']:<10} {params['ccp_alpha']:<10.3f} {accuracy:<10.3f}")
        
        # ============================================================
        # STEP 3: Store Results and Identify Best Model
        # ============================================================
        # Convert results list to DataFrame for easy analysis
        self.results_df = pd.DataFrame(results)
        
        # Display the best model's performance
        print(f"\nBest Model Performance:")
        best_result = self.results_df.loc[self.results_df['accuracy'].idxmax()]
        print(f"Accuracy: {self.best_accuracy:.3f}")
        print(f"Parameters: {dict(best_result.drop(['accuracy', 'model']))}")
        
        # ============================================================
        # STEP 4: Detailed Evaluation of Best Model
        # ============================================================
        print(f"\n" + "="*50)
        print("BEST MODEL DETAILED EVALUATION")
        print("="*50)
        
        # Generate predictions using the best model
        best_predictions = self.best_model.predict(self.X_test)
        
        # Get the original class names (before encoding) for readable output
        target_names = self.target_encoder.classes_.tolist()
        
        # ============================================================
        # STEP 4A: Confusion Matrix
        # ============================================================
        # A confusion matrix shows how many predictions were correct/incorrect for each class
        # Rows = Actual classes, Columns = Predicted classes
        # Diagonal values = Correct predictions, Off-diagonal = Misclassifications
        cm = confusion_matrix(self.y_test, best_predictions)
        print(f"\nConfusion Matrix:")
        
        # Format the confusion matrix for display
        n_classes = len(target_names)
        if n_classes <= 5:  # For 5 or fewer classes, show formatted table
            # Create header row with predicted class names
            header = f"{'Actual':<20} | " + " | ".join([f"{name[:12]:^12}" for name in target_names])
            print(header)
            print("-" * len(header))
            
            # Print each row (actual class) with its predictions
            for i, actual_class in enumerate(target_names):
                row = f"{actual_class[:18]:<20} | "
                row += " | ".join([f"{cm[i,j]:^12}" for j in range(n_classes)])
                print(row)
        else:
            # For many classes (e.g., 10 charge bins), just show the raw matrix
            # This prevents the table from becoming too wide to read
            print(cm)
        
        # ============================================================
        # STEP 4B: Classification Report
        # ============================================================
        # Provides detailed metrics for each class:
        # - Precision: Of all positive predictions, how many were correct?
        # - Recall: Of all actual positives, how many did we find?
        # - F1-Score: Harmonic mean of precision and recall
        # - Support: Number of actual occurrences of each class
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, best_predictions, 
                                   target_names=target_names, zero_division=0))

    def extract_rules(self, max_rules=10):
        """
        Extract decision rules from the best trained model.
        
        Generates human-readable decision rules from the tree structure,
        showing the paths from root to leaf nodes that define predictions.
        
        Args:
            max_rules (int): Maximum number of rules to display (default: 10)
            
        Returns:
            list: List of decision rule strings
        """
        if self.best_model is None:
            raise ValueError("Must call modeling_and_evaluation() first to train a model")
        
        from sklearn import tree as sklearn_tree
        
        print(f"\n" + "="*60)
        print("DECISION RULES FROM BEST MODEL")
        print("="*60)
        
        # Get feature names (accounting for encoded features)
        feature_names = list(self.label_encoders.keys())
        class_names = list(self.target_encoder.classes_)
        
        # Extract rules using tree structure
        # The tree structure is stored in sklearn's internal format
        # Each node has indices for children, features, thresholds, etc.
        rules = []
        tree_structure = self.best_model.tree_
        
        # ============================================================
        # RECURSIVE TREE TRAVERSAL FUNCTION (NESTED FUNCTION / CLOSURE)
        # ============================================================
        # WHY IS THIS FUNCTION DEFINED INSIDE extract_rules()?
        # This is a NESTED FUNCTION (also called a CLOSURE) - a deliberate and
        # well-established design pattern in Python. Here's why this is GOOD PRACTICE:
        #
        # 1. ACCESS TO OUTER SCOPE (Closure Behavior)
        #    - The recurse() function needs access to several variables from extract_rules():
        #      * rules - the list where completed decision rules are collected
        #      * tree_structure - the sklearn tree structure being traversed
        #      * feature_names - for human-readable feature names
        #      * class_names - for human-readable class names
        #    - By defining recurse as a nested function, it automatically has access
        #      to these variables WITHOUT passing them as parameters
        #    - This is called a CLOSURE: the inner function "closes over" the outer
        #      function's variables
        #
        # 2. CLEANER FUNCTION SIGNATURE
        #    Without closure: def recurse(node, depth, rule, rules, tree_structure, feature_names, class_names)
        #    With closure:    def recurse(node, depth, rule)
        #    The closure makes recursive calls much cleaner and easier to read!
        #
        # 3. ENCAPSULATION AND INFORMATION HIDING
        #    - recurse() is only used within extract_rules() and nowhere else
        #    - Keeping it nested prevents it from cluttering the class namespace
        #    - Makes it clear this is a private helper function
        #    - Keeps related code together for better maintainability
        #
        # 4. COMMON PATTERN FOR RECURSIVE ALGORITHMS
        #    This is a standard and elegant pattern for recursive algorithms that need to:
        #    - Maintain shared state (rules list)
        #    - Traverse a data structure (decision tree)
        #    - Build up results incrementally
        #
        # WHEN TO USE NESTED FUNCTIONS:
        # ✅ Inner function is only used by the outer function
        # ✅ Inner function needs access to outer function's local variables
        # ✅ Implementing recursive algorithms with shared state
        # ✅ Want to keep helper functions private and localized
        #
        # This same pattern appears in many standard library implementations
        # and is considered Pythonic and best practice!
        # ============================================================
        
        # ============================================================
        # TREE TRAVERSAL LOGIC
        # ============================================================
        # This inner function performs a depth-first traversal of the decision tree,
        # building up rule strings as it traverses from root to leaves.
        #
        # TREE STRUCTURE EXPLAINED:
        # - Decision trees are binary trees with internal nodes (splits) and leaf nodes (predictions)
        # - Internal nodes test a feature against a threshold and branch left (≤) or right (>)
        # - Leaf nodes contain the final prediction for samples reaching that node
        # - Each path from root to leaf represents a complete decision rule
        #
        # PARAMETERS:
        # - node: Current node index in the tree (integer)
        # - depth: Current depth in the tree (0 = root, increments going deeper)
        # - rule: Accumulated rule string from root to current node (builds up during recursion)
        #
        # EXAMPLE OF RULE BUILDING:
        # Root: "smoker ≤ 0.50"
        # → Left child: "smoker ≤ 0.50 AND age ≤ 35.00"
        #   → Leaf: "smoker ≤ 0.50 AND age ≤ 35.00 → charges_bin_2 (confidence: 85%)"
        # → Right child: "smoker > 0.50 AND bmi > 30.00"
        #   → Leaf: "smoker > 0.50 AND bmi > 30.00 → charges_bin_9 (confidence: 92%)"
        # ============================================================
        
        def recurse(node, depth, rule):
            """
            Recursively traverse the decision tree and extract decision rules.
            
            This function implements depth-first search (DFS) on the tree structure,
            visiting each node and either:
            1. Recording a complete rule at leaf nodes, or
            2. Recursing to both children at internal nodes
            
            Args:
                node (int): Index of current node in tree.tree_ structure
                depth (int): Current depth in tree (0 = root, increases with recursion)
                rule (str): Accumulated rule conditions from root to this node
                           Example: "smoker ≤ 0.50 AND bmi > 28.00"
            
            Returns:
                None (modifies the outer 'rules' list by appending complete rules)
            """
            # Optional: Track indentation for debugging (currently unused)
            indent = "  " * depth
            
            # ============================================================
            # STEP 1: Check Node Type (Leaf vs Internal)
            # ============================================================
            # In sklearn's tree structure, leaf nodes have feature == TREE_UNDEFINED
            # Internal (split) nodes have a valid feature index
            #
            # TREE_UNDEFINED is sklearn's constant (-2) indicating "no feature to split on"
            # This is how we distinguish between:
            # - Leaf nodes: Pure nodes or nodes that meet stopping criteria (no more splits)
            # - Internal nodes: Nodes that still split the data
            
            if tree_structure.feature[node] == sklearn_tree._tree.TREE_UNDEFINED:
                # ========================================================
                # CASE A: LEAF NODE (Terminal Node - Makes Prediction)
                # ========================================================
                # We've reached the end of a decision path.
                # This node contains samples and makes a final prediction.
                
                # Get the class distribution of samples at this leaf
                # tree_structure.value[node] is a 2D array: [n_outputs, n_classes]
                # For single output classification: [1, n_classes]
                # We take [0] to get the class counts: [class_0_count, class_1_count, ...]
                samples = tree_structure.value[node][0]
                
                # Find which class has the most samples (majority vote)
                # np.argmax returns the index of the maximum value
                # This index corresponds to the encoded class label (0, 1, 2, ...)
                predicted_class_idx = np.argmax(samples)
                
                # Convert the numeric class index back to the original class name
                # Example: 5 → "charges_bin_6"
                predicted_class = class_names[predicted_class_idx]
                
                # Calculate prediction confidence (purity of this leaf)
                # Confidence = (samples of predicted class) / (total samples in leaf)
                # Example: If 45 out of 50 samples are "charges_bin_6", confidence = 90%
                # High confidence (>90%) = very pure leaf, strong prediction
                # Low confidence (~50%) = mixed leaf, uncertain prediction
                confidence = samples[predicted_class_idx] / np.sum(samples)
                
                # Format the complete rule as: "conditions → prediction (confidence: %)"
                # Example: "smoker > 0.50 AND bmi > 30.00 → charges_bin_9 (confidence: 92%)"
                rule_str = f"{rule} → {predicted_class} (confidence: {confidence:.2%})"
                
                # Add this complete rule to our collection
                rules.append(rule_str)
                
                # Base case: Leaf node reached, return to explore other branches
                
            else:
                # ========================================================
                # CASE B: INTERNAL NODE (Decision Node - Splits Data)
                # ========================================================
                # This node performs a binary split on a feature.
                # We need to recurse into both left and right children.
                
                # Get the feature used for splitting at this node
                # feature_idx is an integer index into our feature array
                # Example: If features = ['age', 'sex', 'bmi'], feature_idx=2 means 'bmi'
                feature_idx = tree_structure.feature[node]
                
                # Get the threshold value for the split
                # All samples with feature ≤ threshold go left
                # All samples with feature > threshold go right
                # Example: threshold=30.50 means split on "bmi ≤ 30.50" vs "bmi > 30.50"
                threshold = tree_structure.threshold[node]
                
                # Convert feature index to human-readable feature name
                # This makes rules interpretable: "bmi ≤ 30.00" instead of "Feature_2 ≤ 30.00"
                if feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                else:
                    # Fallback for any unexpected feature indices
                    feature_name = f"Feature_{feature_idx}"
                
                # ====================================================
                # BUILD LEFT BRANCH RULE (feature ≤ threshold)
                # ====================================================
                # Samples satisfying this condition go to the left child
                #
                # Rule building logic:
                # - If rule is empty (root node): Start the rule with this condition
                # - If rule exists: Append " AND <condition>" to continue building the path
                #
                # Example progression:
                # Root: "smoker ≤ 0.50" (rule was empty)
                # Child: "smoker ≤ 0.50 AND age ≤ 35.00" (rule existed)
                # Grandchild: "smoker ≤ 0.50 AND age ≤ 35.00 AND bmi ≤ 28.00"
                left_rule = f"{rule} AND {feature_name} ≤ {threshold:.2f}" if rule else f"{feature_name} ≤ {threshold:.2f}"
                
                # Recursively process the left child subtree
                # - tree_structure.children_left[node]: Index of left child node
                # - depth + 1: We're going one level deeper
                # - left_rule: Pass the accumulated rule to the child
                recurse(tree_structure.children_left[node], depth + 1, left_rule)
                
                # ====================================================
                # BUILD RIGHT BRANCH RULE (feature > threshold)
                # ====================================================
                # Samples satisfying this condition go to the right child
                #
                # Same logic as left branch, but with > instead of ≤
                # This ensures we capture both sides of every split
                right_rule = f"{rule} AND {feature_name} > {threshold:.2f}" if rule else f"{feature_name} > {threshold:.2f}"
                
                # Recursively process the right child subtree
                # After this returns, we've explored the entire right subtree
                recurse(tree_structure.children_right[node], depth + 1, right_rule)
                
                # Note: After both left and right recursions complete,
                # we've extracted all rules from this subtree
        
        # Start recursion from root
        recurse(0, 0, "")
        
        # Print and return limited number of rules
        print(f"\nTop {min(max_rules, len(rules))} Rules:\n")
        for i, rule in enumerate(rules[:max_rules], 1):
            print(f"{i}. {rule}")
        
        if len(rules) > max_rules:
            print(f"\n... and {len(rules) - max_rules} more rules")
        
        print(f"\nTotal rules extracted: {len(rules)}")
        
        return rules
    
    def extract_rules_simple(self, max_depth=None):
        """
        Extract decision rules using sklearn's built-in export_text function.
        
        This is a simpler alternative to extract_rules() that uses sklearn's
        built-in text export functionality. Shows tree structure with indentation.
        
        Args:
            max_depth (int): Maximum depth to display (None = show all, default: None)
            
        Returns:
            str: Text representation of the decision tree rules
            
        Example Output:
            |--- smoker <= 0.50
            |   |--- age <= 35.00
            |   |   |--- class: charges_bin_2
            |   |--- age > 35.00
            |   |   |--- class: charges_bin_4
            |--- smoker > 0.50
            |   |--- class: charges_bin_10
        """
        if self.best_model is None:
            raise ValueError("Must call modeling_and_evaluation() first to train a model")
        
        if self.X_train is None:
            raise ValueError("Training data not available. Must call modeling_and_evaluation() first.")
        
        print(f"\n" + "="*60)
        print("DECISION RULES (SIMPLE TEXT FORMAT)")
        print("="*60)
        
        # Get feature names from the training data (includes ALL features used)
        # This is important - we need all feature names, not just the encoded ones
        feature_names = list(self.X_train.columns)
        
        # Use sklearn's built-in export_text - one line does it all!
        tree_rules = export_text(
            self.best_model,
            feature_names=feature_names,
            max_depth=max_depth
        )
        
        print(tree_rules)
        
        if max_depth is not None:
            print(f"\nNote: Display limited to depth {max_depth}")
            print(f"Full tree depth is {self.best_model.get_depth()}")
        
        return tree_rules
    
    def get_aggregate_tbl(self, grp_by='children', agg_by='charges'):
        """
        Get aggregated comparison table across all regions.
        
        Delegates to DataAggregator for actual aggregation logic.
        
        Args:
            grp_by (str): Column to group by (default: 'children')
            agg_by (str): Column to aggregate (default: 'charges')
            
        Returns:
            pd.DataFrame: Pivoted table comparing aggregated values across regions
        """
        return self.aggregator.get_aggregate_table(grp_by, agg_by)

    def get_grpd_bar_chart(self, grp_by='children', agg_by='charges'):
        """
        Generate grouped bar chart comparing regions by aggregated values.
        
        Delegates to DataAggregator for visualization.
        
        Args:
            grp_by (str): Column to group by on x-axis (default: 'children')
            agg_by (str): Column to aggregate for y-axis values (default: 'charges')
            
        Returns:
            matplotlib.pyplot: Configured plot object ready to display with plt.show()
        """
        return self.aggregator.get_grouped_bar_chart(grp_by, agg_by)


    def get_southeast_df(self):
        """
        Get dataframe filtered to Southeast region only.
        
        Returns:
            pd.DataFrame: Insurance data for Southeast region
        """
        return self.southeast_df

    def get_northeast_df(self):
        """
        Get dataframe filtered to Northeast region only.
        
        Returns:
            pd.DataFrame: Insurance data for Northeast region
        """
        return self.northeast_df

    def get_southwest_df(self):
        """
        Get dataframe filtered to Southwest region only.
        
        Returns:
            pd.DataFrame: Insurance data for Southwest region
        """
        return self.southwest_df  # Fixed: was returning southeast_df

    def get_northwest_df(self):
        """
        Get dataframe filtered to Northwest region only.
        
        Returns:
            pd.DataFrame: Insurance data for Northwest region
        """
        return self.northwest_df
    

# Example usage:
if __name__ == "__main__":
    # Initialize the Decision Tree classification loader
    print("=" * 60)
    print("Decision Tree Classification Loader - Insurance Data Analysis")
    print("=" * 60)
    
    # ============================================================
    # TARGET VARIABLE SELECTION: Why charges_group?
    # ============================================================
    print("\n" + "=" * 60)
    print("TARGET VARIABLE SELECTION RATIONALE")
    print("=" * 60)
    print("""
charges_group was selected as the default target variable because:

1. BUSINESS VALUE
   - Insurance companies need to predict and stratify healthcare costs
   - Classifying customers into 10 charge bins enables risk stratification
   - Critical for premium setting, resource allocation, and financial planning

2. FEATURE ALIGNMENT
   - Age: Older individuals have higher medical expenses
   - BMI: Higher BMI correlates with increased health risks
   - Smoker Status: Significant predictor of healthcare utilization
   - Region: Geographic variations in healthcare costs

3. IDEAL FOR DECISION TREES
   - 10 meaningful classes (not too few, not too granular)
   - Decision trees excel at capturing non-linear relationships
   - Can identify complex interactions (e.g., "high BMI + smoker")

4. AVOIDS COMMON PITFALLS
   - No data leakage: charges_group is derived but independent
   - Richer than binary classification (not just "high/low")
   - More practical than continuous regression for business decisions

5. REAL-WORLD APPLICABILITY
   - Classify new applicants into risk tiers for pricing
   - Enable case management for high-risk customers
   - Facilitate budget forecasting with granular predictions

Alternative targets (smoker, age_group, bmi_group):
   - Fewer classes (2-4) limiting model complexity
   - Lower business relevance for insurance operations
   - Less direct connection to financial decision-making
    """)
    print("=" * 60 + "\n")
    
    dt_loader = LoadDecisionTree()  # Use default path
    
    # Test 1: Get regional dataframes and show basic info
    print("\nRegional Data Overview:")
    print("-" * 60)
    southeast_data = dt_loader.get_southeast_df()
    northeast_data = dt_loader.get_northeast_df()
    southwest_data = dt_loader.get_southwest_df()
    northwest_data = dt_loader.get_northwest_df()
    
    print(f"Southeast region records: {len(southeast_data)}")
    print(f"Northeast region records: {len(northeast_data)}")
    print(f"Southwest region records: {len(southwest_data)}")
    print(f"Northwest region records: {len(northwest_data)}")
    
    # Show sample of regional data with age groups
    print("\nSample of Southeast Data (first 5 rows):")
    print("-" * 60)
    print(southeast_data.head())

    print("\nSample of Southwest Data (first 5 rows):")
    print("-" * 60)
    print(southwest_data.head())

    print("\nSample of Northwest Data (first 5 rows):")
    print("-" * 60)
    print(northwest_data.head())

    print("\nSample of Northeast Data (first 5 rows):")
    print("-" * 60)
    print(northeast_data.head())
    
    # Get aggregated table by children
    print("\nAverage Insurance Charges by Number of Children (All Regions):")
    print("-" * 60)
    agg_table = dt_loader.get_aggregate_tbl(grp_by='children', agg_by='charges')
    print(agg_table)
    
    # Get aggregated table by smoker status
    print("\nAverage Insurance Charges by Smoker Status (All Regions):")
    print("-" * 60)
    smoker_table = dt_loader.get_aggregate_tbl(grp_by='smoker', agg_by='charges')
    print(smoker_table)

    # Get aggregated table by age group
    print("\nAverage Insurance Charges by Age Group (All Regions):")
    print("-" * 60)
    age_table = dt_loader.get_aggregate_tbl(grp_by='age_group', agg_by='charges')
    print(age_table)

    # Get aggregated table by bmi group
    print("\nAverage Insurance Charges by BMI Group (All Regions):")
    print("-" * 60)
    agg_table = dt_loader.get_aggregate_tbl(grp_by='bmi_group', agg_by='charges')
    print(agg_table)
    
    # Generate visualization
    print("\nGenerating Bar Chart Visualization...")
    print("-" * 60)
    print("Creating grouped bar chart for charges by children across regions...")
    
    # Uncomment the lines below to display the chart
    # plt_obj = dt_loader.get_grpd_bar_chart(grp_by='children', agg_by='charges')
    # plt_obj.show()

    # - When `df.plot()` or `plt.figure()` is called, matplotlib 
    # creates a __figure object__ in memory
    # - All subsequent `plt` commands (title, xlabel, etc.) operate 
    # on this current figure
    # - `plt.savefig()` saves this current figure to disk
    # - `plt.close()` removes the figure from memory

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create the plot
    dt_loader.get_grpd_bar_chart(grp_by='children', agg_by='charges')
    # Now save it
    plt.savefig('plots/children_charges.png', dpi=150, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print("Plot saved to plots/children_charges.png")

    # Create the plot
    dt_loader.get_grpd_bar_chart(grp_by='smoker', agg_by='charges')
    # Now save it
    plt.savefig('plots/smoker_charges.png', dpi=150, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print("Plot saved to plots/smoker_charges.png")

    # Create the plot
    dt_loader.get_grpd_bar_chart(grp_by='age_group', agg_by='charges')
    # Now save it
    plt.savefig('plots/age_group_charges.png', dpi=150, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print("Plot saved to plots/age_group_charges.png")

        # Create the plot
    dt_loader.get_grpd_bar_chart(grp_by='bmi_group', agg_by='charges')
    # Now save it
    plt.savefig('plots/bmi_group_charges.png', dpi=150, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print("Plot saved to plots/bmi_group_charges.png")

    print("\nNote: Visualization code is commented out by default.")
    print("Uncomment the plt.show() lines in __main__ to display charts.")
    
    # ============================================================
    # Test Decision Tree Modeling and Evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("TESTING DECISION TREE MODELING")
    print("=" * 60)
    print("\nThis section demonstrates the full modeling pipeline:")
    print("1. Data preparation with encoding and train/test split")
    print("2. Training 12 different Decision Tree configurations")
    print("3. Evaluating models with confusion matrix and classification report")
    print("\nDefault target: charges_group (10 risk tiers for healthcare costs)")
    print("=" * 60)
    
    # Run the modeling and evaluation pipeline
    dt_loader.modeling_and_evaluation()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    # Step 5: Rules and Interpretation
    print("\n" + "=" * 60)
    print("DEMONSTRATING BOTH RULE EXTRACTION METHODS")
    print("=" * 60)
    
    # Method 1: Detailed custom extraction with confidence scores
    print("\nMethod 1: Custom rule extraction (with confidence scores)")
    dt_loader.extract_rules(max_rules=10)
    
    # Method 2: Simple sklearn built-in export_text
    print("\nMethod 2: Simple sklearn export_text (tree structure)")
    dt_loader.extract_rules_simple(max_depth=4)
    
    # Visualize the tree using the new DecisionTreeVisualizer class
    from TreeVisualizer import DecisionTreeVisualizer
    
    # Create visualizer instance
    visualizer = DecisionTreeVisualizer(
        model=dt_loader.best_model,
        X_train=dt_loader.X_train,
        label_encoders=dt_loader.label_encoders,
        target_encoder=dt_loader.target_encoder
    )
    
    # Create both visualizations
    visualizer.visualize_sklearn_style(save_path='plots/decision_tree.png')
    visualizer.visualize_rapidminer_style(save_path='plots/decision_tree_decoded.png')
