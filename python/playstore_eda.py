# ============================================
# Google Play Store App Data Analysis
# Step 1–4: Data Loading & Cleaning
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Libraries imported successfully")

# --------------------------------------------
# STEP 1: Load Datasets
# --------------------------------------------
apps = pd.read_csv("../data/apps.csv")
reviews = pd.read_csv("../data/reviews.csv")

print("\n📂 Dataset Loaded")
print("Apps dataset shape:", apps.shape)
print("Reviews dataset shape:", reviews.shape)

# --------------------------------------------
# STEP 2: Initial Inspection
# --------------------------------------------
print("\n🔍 Apps Dataset Preview:")
print(apps.head())

print("\nℹ️ Apps Dataset Info:")
print(apps.info())

# --------------------------------------------
# STEP 3: Handle Missing Ratings
# --------------------------------------------
print("\n❓ Missing Ratings Before:", apps['Rating'].isnull().sum())

apps = apps.dropna(subset=['Rating'])

print("✅ Missing Ratings After:", apps['Rating'].isnull().sum())
print("Apps shape after rating cleanup:", apps.shape)

# --------------------------------------------
# STEP 4: Clean Installs Column
# --------------------------------------------
# Remove corrupted rows (e.g., 'Free' in Installs)
apps = apps[apps['Installs'].str.contains(r'\d', regex=True)]

# Remove '+' and ',' then convert to int
apps['Installs'] = apps['Installs'].str.replace('[+,]', '', regex=True)
apps['Installs'] = apps['Installs'].astype(int)

print("\n✅ Installs column cleaned")
print(apps['Installs'].head())

# --------------------------------------------
# STEP 5: Clean Price Column
# --------------------------------------------
apps['Price'] = apps['Price'].str.replace('$', '', regex=False)
apps['Price'] = apps['Price'].astype(float)

print("\n✅ Price column cleaned")
print(apps['Price'].head())

# --------------------------------------------
# STEP 5.1: Average Rating by Category
# --------------------------------------------
avg_rating_by_category = (
    apps.groupby('Category')['Rating']
    .mean()
    .sort_values(ascending=False)
)

print("\n⭐ Average Rating by Category:")
print(avg_rating_by_category.head(10))


# --------------------------------------------
# STEP 6: Remove Duplicate Apps
# --------------------------------------------
before = apps.shape[0]
apps = apps.drop_duplicates(subset='App', keep='first')
after = apps.shape[0]

print(f"\n🧹 Duplicates removed: {before - after}")
print("Apps shape after duplicate removal:", apps.shape)

# --------------------------------------------
# STEP 6.1: Reviews Dataset Inspection
# --------------------------------------------
print("\n🔍 Reviews Dataset Preview:")
print(reviews.head())

print("\nℹ️ Reviews Dataset Info:")
print(reviews.info())

# --------------------------------------------
# STEP 6.2: Merge Apps & Reviews
# --------------------------------------------
merged = pd.merge(apps, reviews, on='App', how='inner')

print("\n🔗 Merged Dataset Shape:", merged.shape)

# --------------------------------------------
# Sentiment Distribution
# --------------------------------------------
sentiment_counts = merged['Sentiment'].value_counts()

print("\n💬 Sentiment Distribution:")
print(sentiment_counts)

# --------------------------------------------
# Visualization: Rating vs Sentiment
# --------------------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sentiment', y='Rating', data=merged)
plt.title("Rating vs User Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Rating")
plt.show()


# --------------------------------------------
# STEP 7: Final Clean Dataset Check
# --------------------------------------------
print("\n📊 Final Cleaned Dataset Info:")
print(apps.info())

print("\n🎉 Data cleaning completed successfully!")

# --------------------------------------------
# STEP 7.1: App Update Trend
# --------------------------------------------
apps['Last Updated'] = pd.to_datetime(apps['Last Updated'], errors='coerce')

apps['Update Year'] = apps['Last Updated'].dt.year

updates_per_year = apps['Update Year'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.plot(updates_per_year.index, updates_per_year.values, marker='o')
plt.title("App Update Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Number of App Updates")
plt.show()


# --------------------------------------------
# Visualization: Average Rating by Category
# --------------------------------------------
plt.figure(figsize=(12, 6))
sns.barplot(
    x=avg_rating_by_category.index,
    y=avg_rating_by_category.values
)
plt.xticks(rotation=90)
plt.title("Average Rating by App Category")
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()


# --------------------------------------------
# App Count by Category
# --------------------------------------------
app_count_by_category = apps['Category'].value_counts()

print("\n📱 App Count by Category:")
print(app_count_by_category.head(10))


# --------------------------------------------
# Total Installs by Category
# --------------------------------------------
installs_by_category = (
    apps.groupby('Category')['Installs']
    .sum()
    .sort_values(ascending=False)
)

print("\n🚀 Total Installs by Category:")
print(installs_by_category.head(10))


# --------------------------------------------
# Free vs Paid Apps
# --------------------------------------------
type_counts = apps['Type'].value_counts()

print("\n💰 Free vs Paid Apps:")
print(type_counts)


plt.figure(figsize=(6, 6))
type_counts.plot.pie(autopct='%1.1f%%')
plt.title("Free vs Paid Apps Distribution")
plt.ylabel("")
plt.show()

plt.savefig("../visuals/avg_rating_by_category.png", bbox_inches='tight')
