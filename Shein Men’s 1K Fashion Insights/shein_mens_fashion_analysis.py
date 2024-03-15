
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/mnt/data/shein_mens_fashion.csv.xls'
data = pd.read_csv(file_path)

# Distribution of numerical features
numerical_columns = ['sale_price/amount', 'retail_price/amount', 'discount_percentage', 'reviews_count', 'average_rating']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = axes.flatten()
for ax, column in zip(axes, numerical_columns):
    sns.histplot(data=data, x=column, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
if len(numerical_columns) % 2 != 0:
    fig.delaxes(axes[-1])
plt.tight_layout()

# Categorical feature analysis
category_counts = data['category_name'].value_counts()
color_counts = data['color'].value_counts()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[0])
axes[0].set_title('Number of Products per Category')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(x=color_counts.index, y=color_counts.values, ax=axes[1])
axes[1].set_title('Number of Products per Color')
axes[1].set_xlabel('Color')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()

# Price and Discount Analysis
plt.figure(figsize=(12, 6))
scatter = plt.scatter(x=data['retail_price/amount'], y=data['sale_price/amount'], c=data['discount_percentage'], cmap='viridis')
plt.title('Sale Price vs. Retail Price by Discount Percentage')
plt.xlabel('Retail Price')
plt.ylabel('Sale Price')
plt.colorbar(scatter, label='Discount Percentage')
plt.show()

# Reviews and Ratings Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(x=data['average_rating'], y=data['reviews_count'], alpha=0.6)
plt.title('Reviews Count vs. Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Reviews Count')
plt.show()

# SKU Analysis
sku_counts = data.groupby('category_name')['sku'].nunique()
sku_counts_sorted = sku_counts.sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(y=sku_counts_sorted.index, x=sku_counts_sorted.values)
plt.title('Number of SKUs per Category')
plt.xlabel('Number of SKUs')
plt.ylabel('Category')
plt.show()

# Discount Percentage vs. Reviews Count Analysis
plt.figure(figsize=(14, 7))
scatter = plt.scatter(x=data['discount_percentage'], y=data['reviews_count'], c=data['sale_price/amount'], cmap='viridis')
plt.title('Discount Percentage vs. Reviews Count (colored by Sale Price)')
plt.xlabel('Discount Percentage')
plt.ylabel('Reviews Count')
plt.colorbar(scatter, label='Sale Price')
plt.show()

# Price Trends Analysis
price_trends = data.groupby('category_name').agg({
    'sale_price/amount': 'mean',
    'retail_price/amount': 'mean'
}).sort_values(by='sale_price/amount', ascending=False)
price_trends.plot(kind='bar', figsize=(15, 8), title='Average Sale and Retail Prices per Category')
plt.xlabel('Category')
plt.ylabel('Average Price')
plt.xticks(rotation=90)
plt.show()
