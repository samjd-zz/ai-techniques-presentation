#!/usr/bin/env python3
"""
Feature Engineering Module for Insurance Data

This module provides utilities to create engineered features from raw insurance data,
including age groups, BMI categories, charge bins, and index management.

Author: Shawn Jackson Dyck
Date: October 2025

"""

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class FeatureEngineer:
    """
    Handles feature engineering for insurance dataset analysis.
    
    Creates derived features from raw data including:
    - Age groups (young_adult, middle_aged, senior_adult)
    - BMI groups (based on WHO standards)
    - Charge bins (10 equal-width categories)
    - Index column (ID)
    """
    
    def _add_feature_id(self, df):
        """
        Add an ID column to the dataframe and set it as the index.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: New dataframe with ID column as index (starting from 1)
        """
        df = df.copy()
        # Add ID column starting from 1
        df['ID'] = range(1, len(df) + 1)
        df.set_index('ID', inplace=True)  # Set it as index
        return df

    def _add_feature_age_grps(self, df):
        """
        Create age group categories for classification purposes.
        
        Bins ages into distinct categories:
        - young_adult: 18-29 years
        - middle_aged: 30-44 years
        - senior_adult: 45-64 years
        
        Args:
            df (pd.DataFrame): Input dataframe with 'age' column
            
        Returns:
            pd.DataFrame: New dataframe with added 'age_group' feature
        """
        df = df.copy()
        df['age_group'] = 'adult'  # Default

        df.loc[(df['age'] >= 18) & (df['age'] < 30), 'age_group'] = 'young_adult'
        df.loc[(df['age'] >= 30) & (df['age'] < 45), 'age_group'] = 'middle_aged'
        df.loc[(df['age'] >= 45) & (df['age'] <= 64), 'age_group'] = 'senior_adult'

        return df

    def _add_feature_bin_charges(self, df):
        """
        Create charges bins for classification purposes.
        
        Bins non-zero charges into 10 equal-width categories and handles zero charges separately:
        - Free: Zero charges
        - charges_bin_1 to charges_bin_10: Equal-width bins of non-zero charges
        
        Uses sklearn's KBinsDiscretizer with uniform strategy to create equal-width bins,
        ensuring fair representation across the charge range.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'charges' column
            
        Returns:
            pd.DataFrame: New dataframe with added 'charges_group' feature
        """
        df = df.copy()

        # Apply equal width binning to charges attribute
        # Extract non-zero charges for binning
        non_zero_mask = df['charges'] > 0
        non_zero_charges = df.loc[non_zero_mask, 'charges']
        
        if len(non_zero_charges) > 0:
            # Create 10 equal-width bins
            # subsample=None disables subsampling (silences FutureWarning)
            binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=None)
            charge_bins = binner.fit_transform(non_zero_charges.values.reshape(-1, 1)).flatten()
            
            # Create charge groups
            df['charges_group'] = 'Free'  # Default for zero charges
            df.loc[non_zero_mask, 'charges_group'] = [
                f'charges_bin_{int(bin_num)+1}' for bin_num in charge_bins
            ]
        else:
            df['charges_group'] = 'Free'
        
        return df

    def _add_feature_bmi_grps(self, df):
        """
        Create BMI group categories for classification purposes.
        
        Bins BMI values into distinct categories based on WHO standards:
        - Underweight: BMI < 18.5
        - Normal weight: BMI 18.5-24.9
        - Overweight: BMI 25.0-29.9
        - Obese Class I (Moderate): BMI 30.0-34.9
        - Obese Class II (Severe): BMI 35.0-39.9
        - Obese Class III (Very Severe/Morbid): BMI ≥ 40.0
        
        Args:
            df (pd.DataFrame): Input dataframe with 'bmi' column
            
        Returns:
            pd.DataFrame: New dataframe with added 'bmi_group' feature
        """
        df = df.copy()
        df['bmi_group'] = 'Underweight: BMI < 18.5'  # Default

        df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 24.9), 'bmi_group'] = 'Normal weight'
        df.loc[(df['bmi'] >= 25.0) & (df['bmi'] < 29.9), 'bmi_group'] = 'Overweight'
        df.loc[(df['bmi'] >= 30.0) & (df['bmi'] < 34.9), 'bmi_group'] = 'Obese Class I'
        df.loc[(df['bmi'] >= 35.0) & (df['bmi'] < 39.9), 'bmi_group'] = 'Obese Class II'
        df.loc[(df['bmi'] >= 40.0), 'bmi_group'] = 'Obese Class III'

        return df

    def add_all_features(self, df):
        """
        Add all engineered features to the dataframe in one pass.
        
        This is a convenience method that applies all feature engineering
        transformations in the correct order:
        1. Age groups
        2. BMI groups
        3. Index/ID
        4. Charge bins
        
        Args:
            df (pd.DataFrame): Input dataframe with raw insurance data
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features added
        """
        df = self._add_feature_id(df)
        df = self._add_feature_age_grps(df)
        df = self._add_feature_bmi_grps(df)
        df = self._add_feature_bin_charges(df)
        return df


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 60)
    print("Feature Engineer - Testing Module")
    print("=" * 60)
    
    # Load sample data
    df = pd.read_csv('data/insurance.csv')
    
    print(f"\nOriginal DataFrame shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Add all features
    df_engineered = engineer.add_all_features(df)
    
    print(f"\nEngineered DataFrame shape: {df_engineered.shape}")
    print(f"New columns: {list(df_engineered.columns)}")
    
    # Show sample
    print("\nSample of engineered data (first 5 rows):")
    print(df_engineered.head())
    
    print("\n" + "=" * 60)
    print("Feature Engineering Complete!")
    print("=" * 60)
