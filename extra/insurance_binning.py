import pandas as pd
import matplotlib.pyplot as plt
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

# Method 1: Equal-width binning for Age
df['age_bins'] = pd.cut(df['age'], bins=4, labels=['18-28', '29-39', '40-50', '51+'])

# Method 2: Custom bins for Age
age_bins = [0, 30, 40, 50, 100]
age_labels = ['<30', '30-39', '40-49', '50+']
df['age_custom'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

# Method 3: BMI categories (standard medical categories)
bmi_bins = [0, 18.5, 25, 30, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)

# Method 4: Quantile-based binning (equal frequency)
df['charges_quantiles'] = pd.qcut(df['charges'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Age bins vs Average Charges
age_charges = df.groupby('age_custom')['charges'].mean()
axes[0, 0].bar(age_charges.index, age_charges.values, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Average Charges by Age Group', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('Average Charges ($)')
axes[0, 0].tick_params(axis='x', rotation=0)

# Plot 2: BMI category distribution
bmi_counts = df['bmi_category'].value_counts()
axes[0, 1].bar(bmi_counts.index, bmi_counts.values, color='coral', edgecolor='black')
axes[0, 1].set_title('Distribution by BMI Category', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('BMI Category')
axes[0, 1].set_ylabel('Count')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Charges by Smoker and Age Group
smoker_age = df.groupby(['age_custom', 'smoker'])['charges'].mean().unstack()
smoker_age.plot(kind='bar', ax=axes[1, 0], color=['lightgreen', 'salmon'], edgecolor='black')
axes[1, 0].set_title('Average Charges by Age Group and Smoking Status', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Average Charges ($)')
axes[1, 0].legend(title='Smoker')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Histogram with bins
axes[1, 1].hist(df['charges'], bins=5, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Distribution of Charges (5 bins)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Charges ($)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print binning summaries
print("="*60)
print("BINNING SUMMARY")
print("="*60)
print("\n1. Age Groups (Custom Bins):")
print(df['age_custom'].value_counts().sort_index())

print("\n2. BMI Categories:")
print(df['bmi_category'].value_counts())

print("\n3. Average Charges by Age Group:")
print(df.groupby('age_custom')['charges'].agg(['mean', 'count']))

print("\n4. Average Charges by BMI Category:")
print(df.groupby('bmi_category')['charges'].agg(['mean', 'count']))

print("\n5. Charges Quantiles:")
print(df['charges_quantiles'].value_counts().sort_index())