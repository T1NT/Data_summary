#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries


# In[4]:


import pandas as pd


# In[5]:


import seaborn as sns


# In[6]:


df = pd.read_csv("TWO_CENTURIES_OF_UM_RACES.csv")


# In[7]:


# See the data has been imported


# In[8]:


df.head(10)    # It will show you the first 10 rows associated with it


# In[9]:


df.shape    # It shows the amount of (rows, columns)


# In[10]:


df.dtypes   # It shows the types of data that's in the data frame


# In[11]:


# Clean up data 


# In[12]:


# Only want USA races, 50km or 50miles,  2020


# In[13]:


# Step 1 show 5okm or 50miles
# 50km
# 50mi
# 50miles


# In[14]:


df[df['Event distance/length']== '50miles']           # Testing to see 50km or 50miles


# In[15]:


# Combine the 50km,50mi and 50 miles with isin


# In[16]:


df[df['Event distance/length'] .isin(['50miles','50mi','50km'])]   # we've added the 50km and 50 miles 


# In[17]:


# Next the 2020


# In[18]:


df[(df['Event distance/length'] .isin(['50miles','50mi','50km'])) & (df['Year of event'] == 2020)]


# In[19]:


# Next the USA part


# In[20]:


df[df['Event name'] == 'Everglades 50 Mile Ultra Run (USA)'] ['Event name'].str.split('(').str.get(1).str.split(')').str.get(0)          


# In[21]:


# Arrange the USA code 


# In[22]:


df[df['Event name'].str.split('(').str.get(1).str.split(')').str.get(0) == 'USA']       


# In[23]:


# Combine all the filters together


# In[24]:


df[(df['Event distance/length'] .isin(['50miles','50mi','50km'])) & (df['Year of event'] == 2020) & (df['Event name'].str.split('(').str.get(1).str.split(')').str.get(0) == 'USA')]


# In[25]:


# df2 = combined filters


# In[26]:


df2 = df[(df['Event distance/length'] .isin(['50miles','50mi','50km'])) & (df['Year of event'] == 2020) & (df['Event name'].str.split('(').str.get(1).str.split(')').str.get(0) == 'USA')]


# In[27]:


df2.head(10)


# In[28]:


df2.shape


# In[29]:


# Remove the USA from the event name


# In[30]:


df2['Event name'].str.split('(').str.get(0)


# In[31]:


df2['Event name'] = df2['Event name'] .str.split('(').str.get(0)


# In[32]:


df2.size


# In[33]:


df2.head()


# In[34]:


# Clean up the athlete age


# In[35]:


df2['athlete_age'] = 2020 - df2['Athlete year of birth']


# In[36]:


# Remove the h from the athlete perormance


# In[37]:


df2['Athlete performance'] = df2['Athlete performance'].str.split(' ').str.get(0)


# In[44]:


df2.head()


# In[45]:


# Drop some unecessary columns: Athlete club, Athlete country, Athlete year of birth, Athlete age category


# In[43]:


df2.dtypes


# In[42]:


df2 = df2.drop(['Athlete club', 'Athlete country','Athlete year of birth', 'Athlete age category' ], axis = 1)


# In[47]:


# Clean up null values


# In[48]:


df2.isna().sum()         # This will show where the N/A is and it shows it's in the athlete_age column


# In[50]:


df2[df2['athlete_age'].isna() ==1]


# In[52]:


df2 = df2.dropna()    # This is to drop the rows with the na


# In[53]:


df2.isna().sum()    # Check if there is any na


# In[54]:


# Check for duplicate values


# In[55]:


df2 [df2.duplicated() == True]   


# In[ ]:


# Reset index


# In[57]:


df2.reset_index(drop = True)  # A separate column to count the rows from 0 to 25856


# In[ ]:


# Fix data types


# In[58]:


df2['athlete_age'] = df2['athlete_age'].astype(int)   # from float64 to int


# In[60]:


df2['Athlete average speed'] = df2['Athlete average speed'].astype(float)   # from object to float


# In[61]:


df2.dtypes


# In[62]:


df2.head()


# In[65]:


#rename columns


# In[ ]:


#Year of event                  int64
#Event dates                   object
#Event name                    object
#Event distance/length         object
#Event number of finishers      int64
#Athlete performance           object
#Athlete gender                object
#Athlete average speed        float64
#Athlete ID                     int64
#athlete_age                    int64


# In[67]:


df2 = df2.rename(columns = {'Year of event': 'year' ,  
                 
                  'Event dates': 'race_day',
                 
                 'Event name': 'race_name',
    
                 'Event distance/length' : 'race_length',
                             
                'Event number of finishers' : 'race_number_of_finishers' ,
    
                'Athlete performance' : 'athlete_performance' ,
    
                 'Athlete gender' : 'athlete_gender' , 
    
                 'Athlete average speed' : 'athlete_average_speed' ,
        
                  'Athlete ID' : 'athlete_id'          
                            
})


# In[69]:


df2.head()  # To check the change


# In[ ]:


# Reorder the columns


# In[70]:


df3 = df2 [['race_day', 'race_name', 'race_length', 'race_number_of_finishers', 'athlete_id', 'athlete_gender', 'athlete_age', 'athlete_performance', 'athlete_average_speed']]


# In[71]:


df3.head()   # Notice the year columnn isn't there


# In[72]:


# Find 2 races I ran in 2020 - Sarasota/Everglades


# In[75]:


df3[df3['race_name'] == 'Everglades 50 Mile Ultra Run ']


# In[ ]:


#222509


# In[76]:


df3[df3['athlete_id'] == 222509]


# In[ ]:


#Charts and Graphs


# In[79]:


sns.histplot(df3['race_length'])        # comparing 50km vs 50miles


# In[80]:


sns.histplot(df3['athlete_gender'])           # Man vs Woman 


# In[81]:


sns.histplot(df3['athlete_age'])          # The ages


# In[84]:


sns.histplot(df3, x = 'race_length', hue = 'athlete_gender') 

# The hue helps to overlay the athlete_gender on the race_length histogram


# In[87]:


sns.displot(df3[df3['race_length'] == '50mi']['athlete_average_speed'])           
                # Distribbution plot (comparing the 50mile race length to the athlete average speed)


# In[95]:


sns.violinplot(data = df3, x = 'race_length', y = 'athlete_average_speed', hue = 'athlete_gender')
# the violin plot


# In[96]:


sns.violinplot(data = df3, x = 'race_length', y = 'athlete_average_speed', hue = 'athlete_gender', split = True)
#Add split adjustment  {It brings the 50km for m/f together and 50mi m/f together }


# In[97]:


sns.violinplot(data = df3, x = 'race_length', y = 'athlete_average_speed', hue = 'athlete_gender', split = True, inner = 'quarts')
# Add another adjustment {the inner shows the quartiles}


# In[103]:


sns.violinplot(data = df3, x = 'race_length', y = 'athlete_average_speed', hue = 'athlete_gender', split = True, inner = 'quarts', linewidth = 2)
# Add a linewidth change


# In[107]:


sns.lmplot(data = df3, x = 'athlete_age', y = 'athlete_average_speed', hue = 'athlete_gender')
# A linear graph


# In[108]:


# Question I want to find out from the date


# In[111]:


#race_day                     
#race_name                    
#race_length                  
#race_number_of_finishers      
#athlete_id                    
#athlete_gender               
#athlete_age                   
#athlete_performance          
#athlete_average_speed       


# ### Difference in speed for the 50mi, 50km male to female

# In[112]:


df3.groupby(['race_length', 'athlete_gender'])['athlete_average_speed'].mean()


# ### What age groups are the best in the 50m race (20 + race mins) (show 20)

# In[119]:


df3.query('race_length == "50mi"').groupby('athlete_age')['athlete_average_speed'].agg(['mean', 'count']).sort_values('mean', ascending = False).query('count > 19').head(20)


# ### What age groups are the worst in the 50m race (10 + race mins) (show 20)

# In[122]:


df3.query('race_length == "50mi"').groupby('athlete_age')['athlete_average_speed'].agg(['mean', 'count']).sort_values('mean', ascending = True).query('count > 9').head(20)


# ### Seasons for the data -> slower in summer than winter?
# #### Spring 3-5
# #### Summer 6-8
# #### Fall 9-12
# #### Winter 12-2
# ##### Split between 2 decimals

# In[140]:


df3['race_month'] = df3['race_day'].str.split('.').str.get(1).astype(int)


# In[141]:


df3.head()


# In[142]:


df3['race_season'] = df3['race_month'].apply(lambda x : 'Winter' if x > 11 else 'Fall' if x > 8 else 'Summer' if x > 5 else 'Spring' if x > 2 else 'Winter')


# In[143]:


df3.head(25)


# In[144]:


df3.groupby('race_season')['athlete_average_speed'].agg(['mean', 'count']).sort_values('mean', ascending = False)


# In[ ]:


# 50 miles only


# In[146]:


df3.query('race_length == "50mi"').groupby('race_season')['athlete_average_speed'].agg(['mean', 'count']).sort_values('mean', ascending = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




