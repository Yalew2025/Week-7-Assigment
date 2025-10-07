### Week-7-Assigment
Analyzing Data with Pandas and Visualizing Results with Matplotlib


# %% [markdown]
# # Data Analysis with Pandas and Visualization with Matplotlib
# 
# ## Assignment: Analyzing the Iris Dataset
# 
# This notebook demonstrates loading, cleaning, analyzing, and visualizing data using Pandas and Matplotlib.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully!")

# %% [markdown]
# ## Task 1: Load and Explore the Dataset

# %%
# Load the Iris dataset
try:
    # Method 1: Load from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Iris dataset loaded successfully from sklearn!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Alternative: Load from CSV if available
    try:
        df = pd.read_csv('iris.csv')
        print("Dataset loaded from CSV file")
    except:
        print("Could not load dataset from any source")

# %%
# Display first few rows
print("First 10 rows of the dataset:")
print(df.head(10))

# %%
# Explore dataset structure
print("\n Dataset Information:")
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n Column Names:")
print(df.columns.tolist())

print("\n Data Types:")
print(df.dtypes)

# %%
# Check for missing values
print("\n Missing Values Analysis:")
missing_values = df.isnull().sum()
print(missing_values)

# Check if there are any missing values
if missing_values.sum() == 0:
    print(" No missing values found!")
else:
    print(f" Found {missing_values.sum()} missing values")
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward fill for simplicity
    print(" Missing values handled using forward fill")

# %%
# Basic information about the dataset
print("\n Dataset Info:")
df.info()

# %% [markdown]
# ## Task 2: Basic Data Analysis

# %%
# Compute basic statistics for numerical columns
print(" Statistical Summary of Numerical Columns:")
print(df.describe())

# %%
# Additional statistics
print("\n Additional Statistics:")
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if col != 'species':  # Skip species if it's numeric coded
        print(f"\n{col.upper()}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Standard Deviation: {df[col].std():.2f}")
        print(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")

# %%
# Group by species and compute means
print("\n Statistics by Species:")
species_stats = df.groupby('species').agg({
    'sepal length (cm)': ['mean', 'std', 'min', 'max'],
    'sepal width (cm)': ['mean', 'std', 'min', 'max'],
    'petal length (cm)': ['mean', 'std', 'min', 'max'],
    'petal width (cm)': ['mean', 'std', 'min', 'max']
}).round(2)

print(species_stats)

# %%
# Find interesting patterns
print("\n Interesting Findings:")

# Which species has the largest petals?
max_petal_length = df.loc[df['petal length (cm)'].idxmax()]
print(f"1. Largest petal length: {max_petal_length['petal length (cm)']}cm ({max_petal_length['species']})")

# Which species has the smallest sepals?
min_sepal_length = df.loc[df['sepal length (cm)'].idxmin()]
print(f"2. Smallest sepal length: {min_sepal_length['sepal length (cm)']}cm ({min_sepal_length['species']})")

# Correlation between petal length and width
correlation = df['petal length (cm)'].corr(df['petal width (cm)'])
print(f"3. Correlation between petal length and width: {correlation:.3f}")

# Average measurements by species
print("\n4. Average Measurements by Species:")
avg_by_species = df.groupby('species').mean()
print(avg_by_species)

# %% [markdown]
# ## Task 3: Data Visualization

# %%
# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# %% [markdown]
# ### Visualization 1: Line Chart - Feature Trends by Sample Index

# %%
# Create subplot for line chart
plt.subplot(2, 3, 1)

# Since Iris dataset doesn't have time, we'll use sample index as x-axis
# Sort by sepal length to create a meaningful trend
df_sorted = df.sort_values('sepal length (cm)').reset_index(drop=True)

plt.plot(df_sorted.index, df_sorted['sepal length (cm)'], label='Sepal Length', marker='o', markersize=2)
plt.plot(df_sorted.index, df_sorted['petal length (cm)'], label='Petal Length', marker='s', markersize=2)
plt.plot(df_sorted.index, df_sorted['sepal width (cm)'], label='Sepal Width', marker='^', markersize=2)
plt.plot(df_sorted.index, df_sorted['petal width (cm)'], label='Petal Width', marker='d', markersize=2)

plt.title('Feature Trends Across Samples\n(Sorted by Sepal Length)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index (Sorted)')
plt.ylabel('Measurement (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# %% [markdown]
# ### Visualization 2: Bar Chart - Average Measurements by Species

# %%
plt.subplot(2, 3, 2)

# Prepare data for bar chart
species_means = df.groupby('species').mean()
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_pos = np.arange(len(species_means.index))
width = 0.2

for i, feature in enumerate(features):
    plt.bar(x_pos + i*width, species_means[feature], width, label=feature.replace(' (cm)', ''))

plt.title('Average Measurements by Species', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.xticks(x_pos + width*1.5, species_means.index, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')

# %% [markdown]
# ### Visualization 3: Histogram - Distribution of Sepal Length

# %%
plt.subplot(2, 3, 3)

# Create histogram for sepal length distribution
plt.hist([df[df['species']=='setosa']['sepal length (cm)'],
          df[df['species']=='versicolor']['sepal length (cm)'],
          df[df['species']=='virginica']['sepal length (cm)']],
         bins=12, alpha=0.7, label=['setosa', 'versicolor', 'virginica'])

plt.title('Distribution of Sepal Length by Species', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# %% [markdown]
# ### Visualization 4: Scatter Plot - Sepal Length vs Petal Length

# %%
plt.subplot(2, 3, 4)

# Create scatter plot
colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
                species_data['petal length (cm)'],
                c=colors[species], label=species, alpha=0.7, s=60)

plt.title('Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# %% [markdown]
# ### Additional Visualization 5: Box Plot - Feature Distribution by Species

# %%
plt.subplot(2, 3, 5)

# Create box plot for sepal width
box_data = [df[df['species']==species]['sepal width (cm)'] for species in df['species'].unique()]
plt.boxplot(box_data, labels=df['species'].unique())

plt.title('Sepal Width Distribution by Species', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.grid(True, alpha=0.3)

# %% [markdown]
# ### Additional Visualization 6: Correlation Heatmap

# %%
plt.subplot(2, 3, 6)

# Calculate correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Create a simple correlation plot using bars for simplicity
features = correlation_matrix.columns
correlation_with_sepal_length = correlation_matrix['sepal length (cm)']

plt.bar(range(len(correlation_with_sepal_length)), correlation_with_sepal_length.values)
plt.title('Correlation with Sepal Length', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(range(len(features)), [f.replace(' (cm)', '\n') for f in features], rotation=45)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary of Findings

# %%
print("SUMMARY OF FINDINGS")
print("=" * 50)

print("\n1. DATA OVERVIEW:")
print(f"   - Total samples: {len(df)}")
print(f"   - Features: {len(df.columns) - 1} numerical measurements + species label")
print(f"   - Species distribution: {df['species'].value_counts().to_dict()}")

print("\n2.  KEY PATTERNS:")
print("- Setosa species has distinctly smaller petals")
print(" - Virginica generally has the largest measurements")
print(" - Strong correlation between petal length and width")
print("- Sepal width shows least variation across species")

print("\n3. MEASUREMENT RANGES:")
for col in numeric_columns:
    if col != 'species':
        print(f"- {col}: {df[col].min():.1f}cm to {df[col].max():.1f}cm")

print("\n4. VISUALIZATION INSIGHTS:")
print("- Line chart shows clear measurement trends across sorted samples")
print("- Bar chart reveals consistent size patterns across species")
print("- Histogram displays overlapping but distinct distributions")
print("- Scatter plot shows clear clustering by species")

print("\n" + "=" * 50)
print(" Analysis completed successfully!")

# %%
# Save the cleaned dataset 
try:
    df.to_csv('cleaned_iris_dataset.csv', index=False)
    print("\n Cleaned dataset saved as 'cleaned_iris_dataset.csv'")
except Exception as e:
    print(f"\n Could not save dataset: {e}")
