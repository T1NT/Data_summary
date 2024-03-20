#!/usr/bin/env python
# coding: utf-8

# ## Trend Analysis

# In[1]:


import pandas as pd

# Load the datasets
suicide_rates_filepath = 'suicide_rates_1990-2022.csv'
age_std_suicide_rates_filepath = 'age_std_suicide_rates_1990-2022.csv'

suicide_rates = pd.read_csv(suicide_rates_filepath)
age_std_suicide_rates = pd.read_csv(age_std_suicide_rates_filepath)

# Display the first few rows of each dataset to understand their structure
suicide_rates_head = suicide_rates.head()
age_std_suicide_rates_head = age_std_suicide_rates.head()

# Get a summary of each dataset to check the years and data availability
suicide_rates_summary = suicide_rates.describe(include='all')
age_std_suicide_rates_summary = age_std_suicide_rates.describe(include='all')

(suicide_rates_head, suicide_rates_summary, age_std_suicide_rates_head, age_std_suicide_rates_summary)


# In[2]:


import matplotlib.pyplot as plt

# Convert the year column to datetime format for both datasets
suicide_rates['Year'] = pd.to_datetime(suicide_rates['Year'], format='%Y')
age_std_suicide_rates['Year'] = pd.to_datetime(age_std_suicide_rates['Year'], format='%Y')

# Aggregate the data by year for both datasets to get the mean death rates per 100,000 population
annual_crude_rate = suicide_rates.groupby(suicide_rates['Year'])['DeathRatePer100K'].mean()
annual_age_std_rate = age_std_suicide_rates.groupby(age_std_suicide_rates['Year'])['DeathRatePer100K'].mean()

# Plotting the trend over time
plt.figure(figsize=(14, 7))

# Crude rate trend
plt.plot(annual_crude_rate.index, annual_crude_rate, label='Crude Death Rate per 100,000', color='red')

# Age-standardized rate trend
plt.plot(annual_age_std_rate.index, annual_age_std_rate, label='Age-Standardized Death Rate per 100,000', color='blue')

plt.title('Global Suicide Rates Trend (1990-2022)')
plt.xlabel('Year')
plt.ylabel('Death Rate per 100,000 Population')
plt.legend()
plt.grid(True)
plt.show()


# The trend analysis for global suicide rates from 1990 to 2022 shows two distinct lines:
# 
# The red line represents the Crude Death Rate per 100,000 population. This rate fluctuates over time, showing some variations in the number of suicides per 100,000 people in the population, not adjusted for age.
# The blue line represents the Age-Standardized Death Rate per 100,000 population. This rate is generally smoother and more stable over time, indicating that when adjusting for age differences in the population, the suicide rate trends are less volatile.

# In[ ]:





# ## Comparative Analysis by Region or Country

# In[3]:


# Aggregate the data to get the average death rates per 100,000 population by region and country
region_country_crude_rate = suicide_rates.groupby(['RegionName', 'CountryName'])['DeathRatePer100K'].mean().reset_index()

# Sort the data to find regions and countries with the highest and lowest average suicide rates
region_country_crude_rate_sorted = region_country_crude_rate.sort_values(by='DeathRatePer100K', ascending=False)

# Display the top and bottom countries for suicide rates to get an overview
top_countries_crude_rate = region_country_crude_rate_sorted.head(10)
bottom_countries_crude_rate = region_country_crude_rate_sorted.tail(10)

(top_countries_crude_rate, bottom_countries_crude_rate)


# In[4]:


import seaborn as sns

# Increase the size of the plot
plt.figure(figsize=(12, 8))

# Combine top and bottom countries to create a more manageable subset for visualization
top_bottom_countries_crude_rate = pd.concat([top_countries_crude_rate, bottom_countries_crude_rate])

# Create a bar plot for the top and bottom countries in suicide rates
sns.barplot(x='DeathRatePer100K', y='CountryName', data=top_bottom_countries_crude_rate, hue='RegionName', dodge=False)

plt.title('Top and Bottom Countries by Average Suicide Rates (Crude, per 100,000 Population)')
plt.xlabel('Average Suicide Rate (per 100,000 Population)')
plt.ylabel('Country')
plt.legend(title='Region', loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()


# The Comparative Analysis by Region or Country reveals significant variances in average suicide rates across different regions and countries, based on the crude death rate per 100,000 population:
# 
# Top Countries for Suicide Rates
# - Countries like Lithuania, Hungary, and the Russian Federation in Europe have some of the highest average suicide rates, with Lithuania leading at approximately 38.04 per 100,000 population.
# - Republic of Korea in Asia also shows a high suicide rate, indicating that this issue is significant in both European and Asian contexts.
# 
# Lowest Countries for Suicide Rates
# - On the other end of the spectrum, countries like Jamaica, Malaysia, and Jordan report very low average suicide rates, with Egypt and Iraq having notably low rates as well.
# - It is important to note that for some countries like Seychelles, Dominica, and Saint Kitts and Nevis, the rates are extremely low or not available, which could be due to data limitations or genuinely low incidence rates.
# 

# In[ ]:





# ## Socio-Economic Analysis

# In[5]:


# Prepare the data by selecting the necessary columns for socio-economic analysis
socio_economic_data = suicide_rates[['Year', 'CountryName', 'DeathRatePer100K', 'GDPPerCapita', 'GNIPerCapita', 'InflationRate', 'EmploymentPopulationRatio']]

# Calculate the mean of these indicators for each country over the available years to simplify the analysis
socio_economic_data_mean = socio_economic_data.groupby('CountryName').mean()

# Calculate the correlation matrix to see the relationship between suicide rates and economic indicators
correlation_matrix = socio_economic_data_mean.corr()

# Focus on the correlation with 'DeathRatePer100K'
correlation_with_suicide = correlation_matrix['DeathRatePer100K'].sort_values(ascending=False)

correlation_with_suicide


# In[6]:


# Visualize the correlation between suicide rates and economic indicators
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_suicide.index, y=correlation_with_suicide.values)
plt.title('Correlation between Suicide Rates and Economic Indicators')
plt.xlabel('Economic Indicator')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# The Socio-Economic Analysis reveals the following correlations between suicide rates (Death Rate per 100,000 Population) and economic indicators:
# 
# - Inflation Rate: Shows a positive correlation of 0.24 with suicide rates, suggesting that higher inflation may be associated with higher suicide rates.
# 
# - GDP Per Capita: Has a slight positive correlation of 0.09 with suicide rates, indicating that wealthier countries may have slightly higher suicide rates, though this correlation is weak.
# 
# - GNI Per Capita: Shows a negligible correlation with suicide rates (-0.006), implying no significant relationship.
# 
# - Employment Population Ratio: Exhibits a negative correlation of -0.10 with suicide rates, suggesting that higher employment rates might be associated with lower suicide rates, though this relationship is also relatively weak.
# 
# These correlations indicate that economic factors like inflation and employment may influence suicide rates, although the relationships are not very strong. Inflation shows the most significant positive correlation, suggesting that economic instability might be a stress factor contributing to higher suicide rates. However, it's important to remember that correlation does not imply causation, and these factors should be explored further in a detailed analysis.
# 
# 
# 
# 
# 

# The bar chart visualizes the correlation between suicide rates and various economic indicators. The positive correlation with the inflation rate is the most pronounced, while GDP per capita shows a slight positive correlation. The employment population ratio has a slight negative correlation with suicide rates, and the GNI per capita shows almost no correlation. This visualization highlights the nuanced relationship between economic conditions and suicide rates, suggesting areas for further investigation

# In[ ]:





# ## Yearly and Decadal Analysis

# In[7]:


# For the yearly analysis, we already have the annual data aggregated
# We will plot this again to focus on the yearly trends
plt.figure(figsize=(14, 7))
plt.plot(annual_crude_rate.index, annual_crude_rate, label='Crude Death Rate per 100,000', marker='o', linestyle='-')
plt.plot(annual_age_std_rate.index, annual_age_std_rate, label='Age-Standardized Death Rate per 100,000', marker='x', linestyle='-')

plt.title('Yearly Suicide Rates Trend (1990-2022)')
plt.xlabel('Year')
plt.ylabel('Death Rate per 100,000 Population')
plt.legend()
plt.grid(True)
plt.show()

# For the decadal analysis, we need to group the data by decade
# We define each decade based on the year
decade_labels = ['1990s', '2000s', '2010s', '2020s']
suicide_rates['Decade'] = pd.cut(suicide_rates['Year'].dt.year, bins=[1990, 2000, 2010, 2020, 2030], labels=decade_labels, right=False)
age_std_suicide_rates['Decade'] = pd.cut(age_std_suicide_rates['Year'].dt.year, bins=[1990, 2000, 2010, 2020, 2030], labels=decade_labels, right=False)

# Calculate the mean suicide rates for each decade
decadal_crude_rate = suicide_rates.groupby('Decade')['DeathRatePer100K'].mean()
decadal_age_std_rate = age_std_suicide_rates.groupby('Decade')['DeathRatePer100K'].mean()

# Plot the decadal trends
plt.figure(figsize=(14, 7))
plt.bar(decadal_crude_rate.index, decadal_crude_rate, width=0.4, label='Crude Death Rate per 100,000', align='center')
plt.bar(decadal_age_std_rate.index, decadal_age_std_rate, width=0.4, label='Age-Standardized Death Rate per 100,000', align='edge')

plt.title('Decadal Suicide Rates Trend (1990s-2020s)')
plt.xlabel('Decade')
plt.ylabel('Average Death Rate per 100,000 Population')
plt.legend()
plt.grid(True, axis='y')
plt.show()


# The Yearly and Decadal Analysis of suicide rates from 1990 to 2022 provides the following insights:
# 
# Yearly Analysis
# - The line graphs show the yearly trends for both crude and age-standardized suicide rates. There are fluctuations in the crude rates, while age-standardized rates appear more stable over time, highlighting the importance of considering age distribution in suicide rate analyses.
# 
# Decadal Analysis
# - The bar chart illustrates the average suicide rates per decade, comparing crude and age-standardized rates. Both types of rates show trends over the decades, with some variations indicating changes in suicide rates over time.
# 
# This analysis helps to understand both short-term annual fluctuations and long-term decadal trends in global suicide rates, providing a comprehensive view of how these rates have evolved over the past 30+ years

# In[ ]:





# In[ ]:





# Based on the analyses conducted—Trend Analysis Over Time, Comparative Analysis by Region or Country, Socio-Economic Analysis, and Yearly and Decadal Analysis—here are the key insights and recommendations:
# 
# ### Insights
# 1. **Time Trends:**
#    - Suicide rates have shown fluctuations over the years with periods of increase and decrease, indicating the influence of various global and local factors over time.
#    - Age-standardized rates are more stable compared to crude rates, underscoring the importance of considering age distribution in suicide rate analyses.
# 
# 2. **Regional and Country Variations:**
#    - There are significant geographical variations in suicide rates, with countries in certain regions, notably Eastern Europe and parts of Asia, displaying higher rates.
#    - Conversely, some countries, particularly in Africa and the Middle East, reported much lower rates, which could be influenced by cultural, economic, or reporting differences.
# 
# 3. **Socio-Economic Factors:**
#    - Inflation showed a positive correlation with suicide rates, suggesting that economic instability might be a contributing factor to higher suicide incidences.
#    - Employment rates and economic prosperity (as indicated by GDP per capita) had weaker correlations with suicide rates, indicating that other factors also play a critical role in influencing suicide trends.
# 
# 4. **Long-Term Trends:**
#    - Decadal analysis highlighted shifts in suicide rates over longer periods, which could be attributed to changes in societal, economic, and health-related factors globally.
# 
# ### Recommendations
# 1. **Targeted Mental Health Initiatives:**
#    - Focus on regions and countries with higher suicide rates for targeted mental health support and intervention programs.
#    - Implement culturally sensitive and accessible mental health services to address the specific needs of these populations.
# 
# 2. **Economic Stability Programs:**
#    - Develop and promote policies aimed at economic stabilization to mitigate the impact of inflation and economic downturns on mental health.
#    - Support employment and income-generating activities as part of a broader strategy to improve mental well-being and reduce suicide risks.
# 
# 3. **Age-Adjusted Health Policies:**
#    - Utilize age-standardized suicide rates for policy-making to ensure that interventions are appropriately targeted across different age groups.
# 
# 4. **Longitudinal Research:**
#    - Conduct further research to understand the long-term trends and factors contributing to changes in suicide rates, considering both global and local contexts.
#    - Invest in continuous monitoring and analysis of suicide trends to inform timely and effective public health responses.
# 
# 5. **Cross-Sector Collaboration:**
#    - Encourage collaboration between health, economic, and social sectors to address the multifaceted nature of suicide and its determinants.
#    - Promote community-based initiatives that engage various stakeholders in suicide prevention efforts.
# 
# By addressing these insights and following these recommendations, it is possible to develop more effective strategies for reducing suicide rates and improving mental health outcomes globally.

# In[ ]:




