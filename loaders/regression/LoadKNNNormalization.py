#!/usr/bin/env python3

"""
KNN Normalization Data Loader Module    

This module provides a data loader and preprocessor for K-Nearest Neighbors (KNN)
classification tasks, including data normalization and outlier detection.

Author: 80%  Shawn Jackson Dyck - 20% claude-sonnet-4-5-20250929.ai
Date: October 2025

"""

# %matplotlib inline  # Jupyter notebook magic command - commented out for standalone Python script
import os
import matplotlib
matplotlib.use('Agg')  # Set backend FIRST
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class LoadKNNNormalization:
    """
    A data loader and preprocessor for KNN classification with normalization capabilities.
    
    Handles loading smartphone activity data, computing statistics, normalizing features,
    detecting outliers, and preparing train/test splits for machine learning workflows.
    """
    
    # constructor ------------------------------------------------------------#
    def __init__(self, csv_data='data/human-smartphones.csv'):
        """
        Initialize the KNN normalization loader with smartphone activity data.
        
        Args:
            csv_data (str): Path to CSV file containing smartphone sensor data
        """
        # use pandas for CSV reading is best practice, as it handles parsing, headers, types, and edge cases much better than NumPy.
        self.pd_df = pd.read_csv(csv_data)

        # Store column names for reference
        self.columns = self.pd_df.columns.tolist()

        # Get feature columns (exclude subject and Activity)
        self.feature_cols = [i for i, col in enumerate(self.columns) if col not in ['subject', 'Activity']]
        
        # keep self.np_data as a NumPy array for specialized numpy-based operations.
        self.np_data = self.pd_df.values
        
        # Get indices of important columns
        self.subject_idx = self.columns.index('subject')
        self.activity_idx = self.columns.index('Activity')
        subjects = self.pd_df['subject'].values

        # Split into subject groups using numpy indexing
        self.subject_data_5 = self.np_data[(subjects >= 1) & (subjects <= 5)]
        self.subject_data_10 = self.np_data[(subjects >= 6) & (subjects <= 10)]
        self.subject_data_15 = self.np_data[(subjects >= 11) & (subjects <= 15)]
        self.subject_data_20 = self.np_data[(subjects >= 16) & (subjects <= 20)]
        self.subject_data_25 = self.np_data[(subjects >= 21) & (subjects <= 25)]
        self.subject_data_30 = self.np_data[(subjects >= 26) & (subjects <= 30)]
    
    # public methods ---------------------------------------------------------

    def get_subject_group(self, group_num):
        """Get subject group by number (1-6)"""
        # Array of pre-filtered subject groups (subjects 1-5, 6-10, 11-15, 16-20, 21-25, 26-30)
        groups = [self.subject_data_5, self.subject_data_10, self.subject_data_15,
                  self.subject_data_20, self.subject_data_25, self.subject_data_30]
        
        # Return group using 0-based indexing (group_num - 1), or None if out of range
        return groups[group_num - 1] if 1 <= group_num <= 6 else None

    def get_features_by_activity(self, activity_name):
        """Extract feature data for a specific activity using numpy"""
        activity_col = self.pd_df['Activity'].values

        # https://www.programiz.com/python-programming/numpy/boolean-indexing
        # Boolean mask is a numpy array containing truth values (True/False) that correspond to each element in the array.
        # Suppose we have an array named array1.
        # array1 = np.array([12, 24, 16, 21, 32, 29, 7, 15])
        # Now let's create a mask that selects all elements of array1 that are greater than 20.
        # boolean_mask = array1 > 20
        # Here, array1 > 20 creates a boolean mask that evaluates to True for elements that are greater than 20, and False for 
        # elements that are less than or equal to 20.
        # The resulting mask is an array stored in the boolean_mask variable as:
        # [False, True, False, True, True, True, False, False]
        # Vectorized comparison: creates boolean array where True indicates activity match (no loops!)
        mask = activity_col == activity_name

        # https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
        # Fancy indexing allows selecting these specific columns in one operation
        # Two-step indexing: [mask] does boolean indexing on rows (1st dimension),
        # [:, self.feature_cols] slices all filtered rows (:) and uses fancy indexing on columns (2nd dimension)
        return self.np_data[mask][:, self.feature_cols]
    
    def compute_activity_statistics(self, activity_name):
        """
        Compute comprehensive statistics for all features of a specific activity.
        
        Args:
            activity_name (str): Name of activity (e.g., 'WALKING', 'SITTING')
            
        Returns:
            dict: Dictionary containing 'mean', 'std', 'min', and 'max' arrays
                  for all features associated with the specified activity
        """
        features = self.get_features_by_activity(activity_name).astype(float)
        return {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0)
        }

    def normalize_features(self, data=None):
        """
        Normalize/standardize features using z-score normalization (standardization).
        
        Transforms data to have mean=0 and standard deviation=1 using the formula:
        z = (x - μ) / σ
        
        Args:
            data (array-like, optional): Data to normalize. If None, uses all features from dataset
            
        Returns:
            np.ndarray: Normalized feature array with same shape as input
        """
        # Always select only numeric feature columns from pd_df
        if data is None:
            data = self.pd_df.iloc[:, self.feature_cols].values.astype(float)
        else:
            # Ensure this is a numeric numpy array
            data = np.array(data, dtype=float)

        # calculates the mean of each feature (column)
        mean = np.mean(data, axis=0)

        # calculates the standard deviation of each feature
        std = np.std(data, axis=0)
        
        # if any feature has zero standard deviation (all values are the same), 
        # it avoids dividing by zero by setting std to 1 for that feature.
        std[std == 0] = 1

        #  For each feature, subtracts the mean and divides by the standard deviation, 
        # applying: z = (x - μ) / σ, where x is the value, μ is the mean, and σ is the standard deviation
        return (data - mean) / std

    def get_subject_statistics(self, subject_id):
        """
        Get basic statistics (count, mean, std) for a specific subject.
        
        Args:
            subject_id (int): The subject ID to analyze (1-30)
            
        Returns:
            dict: Dictionary with:
                - 'count': number of samples for the subject
                - 'mean': mean of each feature for the subject
                - 'std': standard deviation of each feature for the subject
        """
        # Extract the subject IDs from the relevant column
        subjects = self.np_data[:, self.subject_idx]

        # Create a boolean mask for rows matching the given subject_id
        mask = subjects == subject_id

        # Select only the feature columns for the matching subject
        subject_features = self.np_data[mask][:, self.feature_cols]

        # Return a dictionary with count, mean, and std for the subject's features
        return {
            'count': np.sum(mask),
            'mean': np.mean(subject_features, axis=0),
            'std': np.std(subject_features, axis=0)
        }

    def find_outliers_zscore(self, threshold=3):
        """
        Find outliers in the feature columns using the z-score method.
        A data point is considered an outlier if any of its feature values
        are more than `threshold` standard deviations from the mean.
        Returns the indices of outlier rows.
        """
        # Extract numeric features and cast to float!
        features = self.pd_df.iloc[:, self.feature_cols].values.astype(float)

        # Compute the z-score for each value in each feature column
        z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
        
        # Create a boolean mask: True if any feature in the row has z-score > threshold
        outlier_mask = np.any(z_scores > threshold, axis=1)

        # Return the indices of rows that are outliers
        return np.where(outlier_mask)[0]

    def remove_outliers(self, threshold=3, inplace=True):
        """
        Remove outliers from the dataset using the z-score method.
        
        Args:
            threshold (float): Z-score threshold for outlier detection (default: 3)
            inplace (bool): If True, modifies the internal dataset. If False, returns cleaned data (default: True)
            
        Returns:
            tuple: (cleaned_data, removed_count) if inplace=False, otherwise (None, removed_count)
                   cleaned_data: numpy array with outliers removed
                   removed_count: number of outliers removed
        """
        # Find outlier indices
        outlier_indices = self.find_outliers_zscore(threshold=threshold)
        removed_count = len(outlier_indices)
        
        # Create a boolean mask for rows to keep (inverse of outliers)
        keep_mask = np.ones(len(self.np_data), dtype=bool)
        keep_mask[outlier_indices] = False
        
        if inplace:
            # Update internal data structures
            self.np_data = self.np_data[keep_mask]
            self.pd_df = self.pd_df.iloc[keep_mask].reset_index(drop=True)
            
            # Update subject group data
            subjects = self.pd_df['subject'].values
            self.subject_data_5 = self.np_data[(subjects >= 1) & (subjects <= 5)]
            self.subject_data_10 = self.np_data[(subjects >= 6) & (subjects <= 10)]
            self.subject_data_15 = self.np_data[(subjects >= 11) & (subjects <= 15)]
            self.subject_data_20 = self.np_data[(subjects >= 16) & (subjects <= 20)]
            self.subject_data_25 = self.np_data[(subjects >= 21) & (subjects <= 25)]
            self.subject_data_30 = self.np_data[(subjects >= 26) & (subjects <= 30)]
            
            return None, removed_count
        else:
            # Return cleaned data without modifying internal state
            cleaned_data = self.np_data[keep_mask]
            return cleaned_data, removed_count

    def compute_euclidean_distances(self, sample_idx):
        """
        Compute the Euclidean distance from a specific sample to all other samples.
        
        Essential for K-Nearest Neighbors algorithm - finds similarity between samples
        in high-dimensional feature space.
        
        Args:
            sample_idx (int): Index of the reference sample
            
        Returns:
            np.ndarray: 1D array of Euclidean distances from sample to all other samples
        """
        # Ensure features are float type for math operations
        features = self.np_data[:, self.feature_cols].astype(float)
        sample = features[sample_idx]
        
        # Compute Euclidean distance: sqrt(sum((x_i - y_i)^2))
        distances = np.sqrt(np.sum((features - sample)**2, axis=1))
        return distances

    def correlation_matrix(self, feature_indices=None):
        """
        Compute the Pearson correlation matrix for selected features.
        Each cell (i, j) in the matrix shows the linear correlation between feature i and feature j.
        Returns a square matrix with values between -1 (perfect negative) and 1 (perfect positive).
        """
        if feature_indices is None:
            features = self.np_data[:, self.feature_cols]
        else:
            features = self.np_data[:, feature_indices]
        
        # Transpose so each column is a variable, then compute correlation matrix
        return np.corrcoef(features.T)
    
    def filter_by_threshold(self, column_name, threshold, operator='greater'):
        """
        Filter rows in the data array where the value in column_name meets the threshold condition.
        Supported operators: 'greater', 'less', 'equal'.
        Returns only rows that match the condition.
        """
        # Find the index of the target column
        col_idx = self.columns.index(column_name)

        # Extract the column data
        col_data = self.np_data[:, col_idx]

        # Create a boolean mask based on the operator
        if operator == 'greater':
            mask = col_data > threshold
        elif operator == 'less':
            mask = col_data < threshold
        elif operator == 'equal':
            mask = col_data == threshold
        else:
            # error
            raise ValueError("Operator must be 'greater', 'less', or 'equal'")
        
        # https://how.dev/answers/what-is-boolean-masking-on-numpy-arrays-in-python
        # Use the mask to filter rows
        return self.np_data[mask]

    def sample_balanced(self, n_per_activity, random_seed=None):
        """
        Create a balanced dataset by sampling equal number of observations per activity.
        
        Useful for handling class imbalance in machine learning tasks.
        
        Args:
            n_per_activity (int): Number of samples to draw from each activity
            random_seed (int, optional): Random seed for reproducibility
            
        Returns:
            np.ndarray: Balanced dataset with n_per_activity samples from each activity
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        activities = self.pd_df['Activity'].unique()
        sampled_indices = []
        
        for activity in activities:
            activity_col = self.pd_df['Activity'].values
            activity_indices = np.where(activity_col == activity)[0]
            
            # Sample without replacement if we have enough data, otherwise use all available
            if len(activity_indices) >= n_per_activity:
                sampled = np.random.choice(activity_indices, n_per_activity, replace=False)
            else:
                sampled = activity_indices
            
            sampled_indices.extend(sampled)
        
        return self.np_data[sampled_indices]

    def split_train_test(self, train_ratio=0.8, random_seed=None):
        """
        Split dataset into training and testing sets with random shuffling.
        
        Args:
            train_ratio (float): Proportion of data for training (default: 0.8)
            random_seed (int, optional): Random seed for reproducibility
            
        Returns:
            tuple: (train_data, test_data) as numpy arrays
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        n_samples = self.np_data.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        return self.np_data[train_indices], self.np_data[test_indices]

    def modeling_and_evaluation(self, random_seed=42):
        """
        Test multiple KNN configurations and find the best model.
        
        This method performs comprehensive evaluation by:
        1. Preparing and normalizing the data
        2. Testing multiple k values and distance metrics
        3. Training each KNN model on the training set
        4. Evaluating each model on the test set
        5. Identifying and analyzing the best performing model
        
        Args:
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        print(f"\n" + "="*60)
        print("KNN MODELING & EVALUATION")
        print("="*60)
        
        # ============================================================
        # STEP 1: Prepare and normalize data
        # ============================================================
        # Get normalized features
        normalized_features = self.normalize_features()
        
        # Get activity labels (target variable)
        activities = self.pd_df['Activity'].values
        
        # Encode activity labels to numeric values
        self.activity_encoder = LabelEncoder()
        y_encoded = self.activity_encoder.fit_transform(activities)
        
        # Display target classes
        print(f"\nTarget Activities: {list(self.activity_encoder.classes_)}")
        
        # ============================================================
        # STEP 2: Train-Test Split
        # ============================================================
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, y_encoded, test_size=0.3, 
            random_state=random_seed, stratify=y_encoded
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Display class distribution in training set
        print(f"\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = self.activity_encoder.inverse_transform([cls])[0]
            print(f"  {class_name}: {count} ({count/len(y_train):.1%})")
        
        # ============================================================
        # STEP 3: Define KNN parameter combinations
        # ============================================================
        # Test various k values and distance metrics
        param_combinations = [
            {'n_neighbors': 3, 'metric': 'euclidean', 'weights': 'uniform'},
            {'n_neighbors': 5, 'metric': 'euclidean', 'weights': 'uniform'},
            {'n_neighbors': 5, 'metric': 'euclidean', 'weights': 'distance'},
            {'n_neighbors': 7, 'metric': 'euclidean', 'weights': 'uniform'},
            {'n_neighbors': 7, 'metric': 'euclidean', 'weights': 'distance'},
            {'n_neighbors': 3, 'metric': 'manhattan', 'weights': 'uniform'},
            {'n_neighbors': 5, 'metric': 'manhattan', 'weights': 'uniform'},
            {'n_neighbors': 7, 'metric': 'manhattan', 'weights': 'distance'},
            {'n_neighbors': 9, 'metric': 'euclidean', 'weights': 'uniform'},
            {'n_neighbors': 11, 'metric': 'euclidean', 'weights': 'uniform'},
            {'n_neighbors': 15, 'metric': 'euclidean', 'weights': 'uniform'},
            {'n_neighbors': 20, 'metric': 'euclidean', 'weights': 'uniform'}
        ]
        
        # ============================================================
        # STEP 4: Train and Evaluate Each Configuration
        # ============================================================
        print("\nTesting different KNN configurations...")
        print(f"{'#':<3} {'k':<4} {'Metric':<12} {'Weights':<10} {'Accuracy':<10}")
        print("-" * 45)
        
        results = []
        best_accuracy = 0.0
        best_model = None
        
        for i, params in enumerate(param_combinations, 1):
            # Create KNN model with current parameters
            model = KNeighborsClassifier(
                n_neighbors=params['n_neighbors'],
                metric=params['metric'],
                weights=params['weights']
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            result = params.copy()
            result['accuracy'] = accuracy
            result['model'] = model
            results.append(result)
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                self.best_accuracy = best_accuracy
                self.best_model = best_model
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
            
            # Print results
            print(f"{i:<3} {params['n_neighbors']:<4} {params['metric']:<12} "
                  f"{params['weights']:<10} {accuracy:<10.3f}")
        
        # ============================================================
        # STEP 5: Store and Display Results
        # ============================================================
        self.results_df = pd.DataFrame(results)
        
        # Display best model
        print(f"\nBest Model Performance:")
        best_result = self.results_df.loc[self.results_df['accuracy'].idxmax()]
        print(f"Accuracy: {best_accuracy:.3f}")
        print(f"Parameters:")
        print(f"  k (n_neighbors): {int(best_result['n_neighbors'])}")
        print(f"  Distance Metric: {best_result['metric']}")
        print(f"  Weights: {best_result['weights']}")
        
        # ============================================================
        # STEP 6: Detailed Evaluation of Best Model
        # ============================================================
        print(f"\n" + "="*60)
        print("BEST MODEL DETAILED EVALUATION")
        print("="*60)
        
        # Get predictions from best model
        best_predictions = self.best_model.predict(self.X_test)
        
        # Get activity names for readability
        activity_names = list(self.activity_encoder.classes_)
        
        # ============================================================
        # STEP 6A: Confusion Matrix
        # ============================================================
        cm = confusion_matrix(self.y_test, best_predictions)
        print(f"\nConfusion Matrix:")
        
        n_classes = len(activity_names)
        if n_classes <= 10:
            # Format as table for reasonable number of classes
            header = f"{'Actual':<15} | " + " | ".join([f"{name[:8]:^8}" for name in activity_names])
            print(header)
            print("-" * len(header))
            
            for i, actual_class in enumerate(activity_names):
                row = f"{actual_class[:13]:<15} | "
                row += " | ".join([f"{cm[i,j]:^8}" for j in range(n_classes)])
                print(row)
        else:
            # Show raw matrix for many classes
            print(cm)
        
        # ============================================================
        # STEP 6B: Classification Report
        # ============================================================
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, best_predictions, 
                                   target_names=activity_names, zero_division=0))

    def plotActivity(self):
        """
        Visualize mean feature values across different activities.
        
        Plots the first 5 features to show how sensor measurements vary
        between activities like walking, sitting, standing, etc.
        """
        # Extract feature columns and activity labels
        feature_data = self.pd_df.iloc[:, self.feature_cols].astype(float)
        
        # Vectorized groupby operation for efficient aggregation
        # pandas groupby splits data by Activity, applies mean() to each group, 
        # and combines results - all in a single C-optimized pass through the data

        # RESULT: Returns DataFrame with Activity as index, features as columns, values are means
        feature_means = feature_data.groupby(self.pd_df['Activity']).mean()
        
        plt.figure(figsize=(10, 6))
        for i in range(5):  # First 5 features
            col_name = self.columns[self.feature_cols[i]]
            plt.plot(feature_means.index, feature_means.iloc[:, i], 
                    label=col_name, marker='o')
        
        plt.xlabel("Activity")
        plt.ylabel("Mean Feature Value")
        plt.title("Mean of First 5 Features by Activity")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()

        plt.savefig('plots/mean.png', dpi=150, bbox_inches='tight')
        plt.close() # Close the figure to free memory
        print("Plot saved to plots/mean.png")

    def plotFeaturesScatter(self):
        """
        Create a scatter plot of the first two features colored by activity type.
        
        Useful for visualizing feature separability and cluster patterns
        in the data for different activities.
        """
        # Load features for all samples, ensure they're float
        features_df = self.pd_df.copy()
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=features_df.iloc[:, self.feature_cols[0]],
                        y=features_df.iloc[:, self.feature_cols[1]],
                        hue=features_df['Activity'],
                        palette='tab10')
        plt.xlabel(self.columns[self.feature_cols[0]])
        plt.ylabel(self.columns[self.feature_cols[1]])
        plt.title("Scatter Plot of First Two Features by Activity")
        #plt.show()

        plt.savefig('plots/first_two_features.png', dpi=150, bbox_inches='tight')
        plt.close() # Close the figure to free memory
        print("Plot saved to plots/first_two_features.png")


# Example usage:
if __name__ == "__main__":
    # Initialize dataset
    knn_loader = LoadKNNNormalization('data/human-smartphones.csv')
    
    print("="*60)
    print("INITIAL DATA ANALYSIS")
    print("="*60)
    print(f"Initial dataset shape: {knn_loader.np_data.shape}")
    
    # Get statistics for walking activity
    walking_stats = knn_loader.compute_activity_statistics('WALKING')
    print("\nWalking activity - Mean of first 5 features:")
    print(walking_stats['mean'][:5])
    
    # Find and remove outliers
    print("\n" + "="*60)
    print("OUTLIER DETECTION AND REMOVAL")
    print("="*60)
    outlier_indices = knn_loader.find_outliers_zscore(threshold=3)
    print(f"Number of outliers detected: {len(outlier_indices)}")
    print(f"Percentage of outliers: {len(outlier_indices)/len(knn_loader.np_data)*100:.2f}%")
    
    # Remove outliers (inplace)
    _, removed_count = knn_loader.remove_outliers(threshold=3, inplace=True)
    print(f"\nOutliers removed: {removed_count}")
    print(f"Cleaned dataset shape: {knn_loader.np_data.shape}")
    
    # Normalize all features (after outlier removal)
    normalized_data = knn_loader.normalize_features()
    print(f"\nNormalized data shape: {normalized_data.shape}")
    
    # Get balanced sample
    balanced_sample = knn_loader.sample_balanced(n_per_activity=100)
    print(f"\nBalanced sample shape: {balanced_sample.shape}")
    
    # Split data
    train, test = knn_loader.split_train_test(train_ratio=0.8)
    print(f"\nTrain shape: {train.shape}, Test shape: {test.shape}")
    
    # Call KNN Modeling & Evaluation (uses cleaned data)
    knn_loader.modeling_and_evaluation(random_seed=42)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    # Plot mean feature values by activity
    print("\nCreating activity feature means plot...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    knn_loader.plotActivity()  # Saves to mean.png
    
    # Plot scatter of first two features
    print("\nCreating feature scatter plot...")
    knn_loader.plotFeaturesScatter()  # Saves to first_two_features.png
    
    print("\n" + "=" * 60)
    print("Analysis Complete! Plots saved:")
    print("  - mean.png")
    print("  - first_two_features.png")
    print("=" * 60)
