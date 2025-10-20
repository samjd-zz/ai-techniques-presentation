import pandas as pd
import numpy as np

# Sample data
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23, 56, 30, 60],
    'sex': ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 
            'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female'],
    'bmi': [27.9, 33.77, 33, 22.705, 28.88, 25.74, 33.44, 27.74, 29.83, 25.84, 26.22, 26.29, 
            34.4, 39.82, 42.13, 24.6, 30.78, 23.845, 40.3, 35.3, 36.005],
    'children': [0, 1, 3, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 
               'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'no'],
    'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest', 'southeast', 
               'southeast', 'northwest', 'northeast', 'northwest', 'northeast', 'southeast', 
               'southwest', 'southeast', 'southeast', 'southwest', 'northeast', 'northeast', 
               'southwest', 'southwest', 'northeast'],
    'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 
                7281.5056, 6406.4107, 28923.13692, 2721.3208, 27808.7251, 1826.843, 11090.7178, 
                39611.7577, 1837.237, 10797.3362, 2395.17155, 10602.385, 36837.467, 13228.84695]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df.head())
print(f"\nOriginal shape: {df.shape}")
print("\n" + "="*80 + "\n")

# Method 1: Using pd.get_dummies() - Simple and most common
print("METHOD 1: Using pd.get_dummies()")
print("-" * 80)

df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=False)
print(df_encoded.head())
print(f"\nShape after encoding: {df_encoded.shape}")
print("\n" + "="*80 + "\n")

# Method 2: Using pd.get_dummies() with drop_first=True (avoids multicollinearity)
print("METHOD 2: Using pd.get_dummies() with drop_first=True")
print("-" * 80)

df_encoded_drop = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print(df_encoded_drop.head())
print(f"\nShape after encoding: {df_encoded_drop.shape}")
print("\n" + "="*80 + "\n")

# Method 3: Manual one-hot encoding for specific columns
print("METHOD 3: Manual encoding for specific columns")
print("-" * 80)

df_manual = df.copy()

# Encode sex
df_manual['sex_male'] = (df_manual['sex'] == 'male').astype(int)
df_manual['sex_female'] = (df_manual['sex'] == 'female').astype(int)

# Encode smoker
df_manual['smoker_yes'] = (df_manual['smoker'] == 'yes').astype(int)
df_manual['smoker_no'] = (df_manual['smoker'] == 'no').astype(int)

# Encode region
for region in df['region'].unique():
    df_manual[f'region_{region}'] = (df_manual['region'] == region).astype(int)

# Drop original categorical columns
df_manual = df_manual.drop(['sex', 'smoker', 'region'], axis=1)

print(df_manual.head())
print(f"\nShape after encoding: {df_manual.shape}")
print("\n" + "="*80 + "\n")

# Method 4: Using sklearn's OneHotEncoder
print("METHOD 4: Using sklearn OneHotEncoder")
print("-" * 80)

from sklearn.preprocessing import OneHotEncoder

# Select categorical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children', 'charges']

# Create encoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Fit and transform
encoded_array = encoder.fit_transform(df[categorical_cols])

# Create dataframe with encoded columns
encoded_df = pd.DataFrame(
    encoded_array, 
    columns=encoder.get_feature_names_out(categorical_cols)
)

# Combine with numerical columns
df_sklearn = pd.concat([df[numerical_cols].reset_index(drop=True), encoded_df], axis=1)

print(df_sklearn.head())
print(f"\nShape after encoding: {df_sklearn.shape}")
print("\nEncoded column names:")
print(encoder.get_feature_names_out(categorical_cols))
print("\n" + "="*80 + "\n")

# Summary of categorical columns
print("SUMMARY OF CATEGORICAL VARIABLES:")
print("-" * 80)
print("\nSex distribution:")
print(df['sex'].value_counts())
print("\nSmoker distribution:")
print(df['smoker'].value_counts())
print("\nRegion distribution:")
print(df['region'].value_counts())
print("\n" + "="*80)

# Save to CSV example
print("\nTo save the encoded dataset:")
print("df_encoded.to_csv('insurance_encoded.csv', index=False)")