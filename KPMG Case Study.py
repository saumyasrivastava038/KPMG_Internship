#!/usr/bin/env python
# coding: utf-8

# ## Step I: Data Quality Assessment

# In[251]:


import pandas as pd
import numpy as np


# In[252]:


pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',50)


# #### Transaction Data

# In[253]:


# Reading the data
transaction = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx',sheet_name='Transactions',skiprows=1)


# In[254]:


# Looking first 10 rows
transaction.head(10)


# In[255]:


# Looking bottom 5 rows
transaction.tail()


# In[256]:


transaction.info()


# Note that there are missing values in some columns and the last column is not in required format i.e. date

# In[257]:


# Checking the number of missing values in each column
pd.isnull(transaction).sum()


# We note that if value is missing in column 'brand' than it is also missing in columns 'product_line', 'product_class', 'product_size', 'standard_cost' and 'product_first_sold_date'. <br>

# In[258]:


transaction[transaction.brand.isnull()]


# In[259]:


# Deleting the above rows
transaction = transaction[~transaction.brand.isnull()]


# In[260]:


pd.isnull(transaction).sum()


# In[261]:


transaction.online_order.value_counts()


# In[262]:


# Deleting the rows where online_order has null
transaction = transaction[~transaction.online_order.isnull()]


# In[263]:


transaction.info()


# In[264]:


transaction['order_status'].value_counts().sort_values()


# In[265]:


# Taking only those orders which get approved
transaction = transaction[transaction.order_status=='Approved']


# In[266]:


transaction['brand'].value_counts().sort_values()


# In[267]:


transaction['product_line'].value_counts().sort_values()


# In[268]:


transaction['product_class'].value_counts().sort_values()


# In[269]:


transaction['product_size'].value_counts().sort_values()


# Now we are converting the column 'product_first_order_date' into proper format of date

# In[270]:


def to_date(num):
    startdate = "01/01/1900"
    date=[]
    for val in num:
        date.append(pd.to_datetime(startdate) + pd.DateOffset(days=int(val)))
    
    return pd.array(date)


# In[271]:


transaction["Product_First_Sold_Date"] = to_date(transaction["product_first_sold_date"])

# Dropping the original column 
transaction.drop('product_first_sold_date',axis=1,inplace=True)
transaction.head()


# In[272]:


transaction.describe(include='all')


# In[273]:


transaction.info()


# In[274]:


# Number of unique customer ids
transaction.customer_id.nunique()


# #### Customer Demographic Data

# In[275]:


# Reading the data
customerD = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx',sheet_name='CustomerDemographic',skiprows=1)


# In[276]:


# Looking for top 10 rows
customerD.head(10)


# In[277]:


customerD.info()


# In[278]:


# Looking for null values in each columns
pd.isnull(customerD).sum()


# In[279]:


# Dropping invalid column
customerD.drop('default',axis=1,inplace=True)


# In[280]:


# Replacing NaN with empty
customerD['last_name'] = customerD['last_name'].replace(np.NaN, "")


# In[281]:


customerD['gender'].value_counts()


# In[282]:


# Correcting the values in gender column
customerD['gender'] = customerD['gender'].replace('F', 'Female')
customerD['gender'] = customerD['gender'].replace('Femal', 'Female')
customerD['gender'] = customerD['gender'].replace('M', 'Male')


# In[283]:


customerD['gender'].value_counts()


# In[284]:


# Dropping those entries that have DOB missing
customerD = customerD[~customerD.DOB.isnull()]


# In[285]:


# Converting the column into date
customerD['DOB'] = customerD['DOB'].dt.date

# Creating a function to calculate the age
from datetime import date

def calculate_age(born):
    
    today = date.today()
    return(today.year - born.year - ((today.month, today.day) < (born.month, born.day)))


# In[286]:


# Creating new column 'Age'
customerD['Age'] = customerD['DOB'].apply(calculate_age)


# In[287]:


# Converting back to datetime
customerD['DOB'] = pd.to_datetime(customerD['DOB'])


# In[288]:


# Replacing missing job title with mode 
customerD['job_title'] = customerD.job_title.replace(np.NaN, 'Tax Accountant')


# In[289]:


# Replacing missing job category with mode
customerD['job_industry_category'] = customerD.job_industry_category.replace(np.NaN, 'Manufacturing')


# In[290]:


# Now there is no missing values in the data
pd.isnull(customerD).sum()


# In[291]:


customerD['wealth_segment'].value_counts().sort_values()


# In[292]:


customerD['deceased_indicator'].value_counts().sort_values()


# In[293]:


# Removing rows where deceased=Y (count is only 2)
customerD = customerD[customerD.deceased_indicator!='Y']


# In[294]:


customerD['owns_car'].value_counts().sort_values()


# In[295]:


customerD.describe(include='all')


# In[296]:


customerD.info()


# In[297]:


# Counting the number of unique customers
customerD.customer_id.nunique()


# #### Customer Address Data

# In[298]:


# Reading the data
customerA = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx',sheet_name='CustomerAddress',skiprows=1)


# In[299]:


# Looking top 10 rows
customerA.head(10)


# In[300]:


# Looking bottom rows
customerA.tail()


# In[301]:


customerA.info()


# In[302]:


customerA.state.value_counts().sort_values()


# In[303]:


customerA['state'] = customerA['state'].replace('Victoria','VIC')
customerA['state'] = customerA['state'].replace('New South Wales','NSW')


# In[304]:


customerA.state.value_counts().sort_values()


# In[305]:


customerA.country.value_counts().sort_values()


# In[306]:


customerA.describe(include='all')


# In[307]:


# Counting the number of unique customers
customerA.customer_id.nunique()


# #### Merging of data
# Joining customer address and customer demographic data into one 

# In[308]:


customer_data = pd.merge(customerD,customerA,how='left',on='customer_id')


# In[309]:


# Looking top 10 rows
customer_data.head(10)


# In[310]:


# Looking bottom rows
customer_data.tail()


# In[311]:


customer_data.info()


# In[312]:


customer_data[customer_data.state.isnull()]


# In[313]:


# Removing the rows where data is null
customer_data = customer_data[~customer_data.state.isnull()]


# In[314]:


customer_data.info()


# In[315]:


# Count of unique customers
customer_data.customer_id.nunique()


# #### Merging customer data with their transactions data

# In[340]:


df = pd.merge(customer_data,transaction,how='left',on='customer_id')


# In[341]:


# Looking top rows
df.head()


# In[342]:


df.info()


# In[343]:


# Counting of unique customers
df.customer_id.nunique()


# In[344]:


pd.isnull(df).sum()


# In[345]:


# Removing the data where null values are present
df = df[~df.brand.isnull()]


# In[346]:


df.columns


# In[347]:


# Rearranging of columns

df = df[['transaction_id', 'product_id', 'customer_id', 'first_name', 'last_name', 'gender', 'DOB','Age', 'job_title', 
         'job_industry_category', 'past_3_years_bike_related_purchases', 'wealth_segment', 'deceased_indicator', 'owns_car', 
         'address', 'postcode', 'state', 'country', 'tenure', 'property_valuation', 'transaction_date', 'online_order', 
         'order_status', 'brand', 'product_line', 'product_class', 'product_size', 'list_price', 'standard_cost', 
         'Product_First_Sold_Date']]


# In[348]:


df.Age.describe()


# In[349]:


# Removing the customer whose age is 176 years
df = df[df.Age!=176]


# In[350]:


df.Age.describe()


# In[351]:


# Creating a new column 'Age_Bin'
df['Age_Bin'] = pd.cut(df.Age, bins=[0,21,31,41,51,61,71,81,91], labels=[10,20,30,40,50,60,70,80])


# In[352]:


# Creating a new column 'Profit'
df['Profit'] = df.list_price - df.standard_cost


# In[353]:


# Creating a new column 'month' from transaction_date column
df['transaction_month'] = df.transaction_date.dt.month

# Replacing numeric value with month names
df['transaction_month'] = df.transaction_month.replace((1,2,3,4,5,6,7,8,9,10,11,12),('Jan','Feb','Mar','Apr','May','Jun','Jul',
                                                                                    'Aug','Sep','Oct','Nov','Dec'))


# In[354]:


# Creating a new column 'weekday' from transaction_date
df['transaction_weekday'] = df.transaction_date.dt.weekday

# Replacing numeric value with weekdays name
df['transaction_weekday'] = df.transaction_weekday.replace((0,1,2,3,4,5,6),('Mon','Tue','Wed','Thur','Fri','Sat','Sun'))


# In[355]:


df.head()


# In[356]:


df.describe(include='all')


# In[357]:


df.info()


# In[358]:


# Counting of unique customers
df.customer_id.nunique()


# ## Step II: Data Insights

# ### A. Data Exploration

# In[359]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[360]:


# Looking distribution w.r.t. to PROFIT

plt.figure(figsize=(30,30))
col = ['online_order','owns_car','gender','product_line', 'product_class', 'product_size', 
       'transaction_month','transaction_weekday','state','brand','Age_Bin','wealth_segment']

for i in range(12):
    plt.subplot(4,3,i+1)
    df.groupby(col[i])['Profit'].mean().plot(kind='bar',color='brown')
    plt.ylabel(f"Average Profit w.r.t. {col[i]}",size=18)
    plt.xlabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=18)


# ##### Observations:
# 
# 1. Online Order - Average profit for both online and offline is almost same. 
# 2. Owns car - Those who owns cars or not their average profit is around same.
# 3. Gender - The gender **U** have slightly higher profits compared to men and women.
# 4. Product line - **Touring** line on an average have higher average profit as compared to other lines.
# 5. Product class - **Medium** class has higher average profit while high and low have almost same.
# 6. Product size - Product size **large** has higher average profit as compared to other sizes.
# 7. Month - All the months have around same average profit.
# 8. Weekday - All weekdays have almost same average profit. 
# 9. State - All the three states NSW, QLD and VIC have same average profit.
# 10. Brand - WEAREA2B has highest average profit while NORCO bicycles has lowest average profit.
# 11. Age Bin - All the age groups have around same average profit.
# 12. Wealth Segment - All the three wealth segment have around same average profit.

# In[402]:


# Looking for distribution w.r.t. to ONLINE ORDER

plt.figure(figsize=(30,30))
col = ['owns_car','gender','Age_Bin','product_line', 'product_class', 'product_size',
       'transaction_month','transaction_weekday','state','brand','wealth_segment']

for i in range(11):
    plt.subplot(4,3,i+1)
    df.groupby(col[i])['online_order'].sum().plot(kind='bar',color='y')
    plt.ylabel(f"Count of Online Order w.r.t. {col[i]}",size=15)
    plt.xlabel('')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=18)


# ##### Observations:
# 
# 1. Standard product line receives more online order as compared to others.
# 2. Product class medium got more online orders while high and low got the same.
# 3. Product size medium got more online orders as compared to others.
# 4. Age bin - People in the age group of 40-50 years order more online.
# 5. New South Wales(NSW) peoples order more online.
# 6. Solex brand got most of their orders online while rest of the brands got around same online order.
# 7. Mass wealth segment customer order more online as compared to others.

# In[339]:


# Saving locally
# df.to_excel('KPMG_merge_data.xlsx')


# ### B. Model Development
# Using the technique of RFM(recency,frequency,monetary)

# #### Monetary

# In[362]:


# Finding total profit per customer
monetary = df.groupby("customer_id").Profit.sum()
monetary = monetary.reset_index()
monetary.columns = ['customer_id','Profit']
monetary.head()


# #### Frequency

# In[363]:


# Finding total number of transaction per customer
freq = df.groupby("customer_id").transaction_id.count()
freq = freq.reset_index()
freq.columns = ['customer_id','Frequency']
freq.head()


# #### Recency

# In[364]:


# Filtering data for customerid and transaction_date
recency  = df[['customer_id','transaction_date']]

# Finding max data
maximum = max(recency.transaction_date)

# Adding one more day to the max data, so that the max date will have 1 as the difference and not zero.
maximum = maximum + pd.DateOffset(days=1)
recency['diff'] = maximum - recency.transaction_date
recency.head()


# In[365]:


# Dataframe merging by recency
recency1 = pd.DataFrame(recency.groupby('customer_id').diff.min())
recency1 = recency1.reset_index()
recency1.columns = ["customer_id", "Recency"]
recency1.head()


# In[366]:


#Combining all recency, frequency and monetary parameters
RFM = freq.merge(monetary, on = "customer_id")
RFM = RFM.merge(recency1, on = "customer_id")
RFM.head()


# In[367]:


df.columns


# In[368]:


# A new dataframe which contains info of customers 
new_df = df[['customer_id', 'first_name', 'last_name', 'gender', 'DOB', 'Age', 'Age_Bin','job_title', 'job_industry_category', 
             'past_3_years_bike_related_purchases','tenure', 'property_valuation','wealth_segment', 'deceased_indicator', 
             'owns_car','address','postcode', 'state', 'country']]


# In[369]:


new_df = new_df.groupby('customer_id').max()
new_df.head()


# In[370]:


new_df.reset_index(inplace=True)
new_df.head()


# In[371]:


RFM = RFM.merge(new_df,how='left',on='customer_id')


# In[372]:


RFM.head()


# In[373]:


RFM.info()


# In[374]:


# Converting recency to numeric
RFM.Recency = RFM.Recency.dt.days

RFM.head(10)


# In[375]:


RFM.describe(percentiles=[0.25,0.3,0.5,0.6,0.75,0.9,0.95,0.99])


# In[376]:


# Checking the correlation between the variables
correlation = RFM[['Frequency','Profit','Recency','Age','tenure','property_valuation','past_3_years_bike_related_purchases']].corr().round(3)
correlation.style.bar(color=['red','green'],align='zero')


# In[377]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[378]:


# Using these columns to segment and identify top 1000 customers
RFM1 = RFM[['Frequency','Profit','Recency']]
RFM1.head()


# In[379]:


# Using min-max scaler to transform the variables and scale down to same level
RFM1 = scaler.fit_transform(RFM1)
RFM1 = pd.DataFrame(RFM1)
RFM1.columns = ['Frequency','Profit','Recency']
RFM1.head()


# In[380]:


# Creating a new column 'Score' which is weighted sum of 5 columns
# Initially the weights are equal (sum of weights = 1)
# You can adjust the weights according to your requirements

RFM1['Score'] = (0.33*RFM1.Frequency) + (0.33*RFM1.Profit) + 0.34*(1-RFM1.Recency)
RFM1.head(10)


# In[381]:


RFM1.describe(percentiles=[0.25,0.5,0.7,0.75,0.9,0.95])


# In[382]:


# Finally getting the top 1000 customers by cutting the score at 70 percentile.
RFM1['Category'] = pd.cut(RFM1.Score, bins=[0,np.percentile(RFM1.Score,70.65),1],labels=[0,1])
RFM1.head()


# In[383]:


# Concateing the dataframes to get all others info about customers
RFM_final = pd.concat((RFM,RFM1[['Score','Category']]),axis=1)
RFM_final.head()


# In[384]:


# Checking the count of 1s and 0s
RFM_final.Category.value_counts()


# In[401]:


# Saving the data locally
RFM_final.to_excel('KPMG_mine_solution.xlsx')


# ### C. Interpretation of result

# In[385]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[386]:


RFM_final.groupby('Category')['Frequency','Recency','Profit'].mean().round(0)


# In[419]:


plt.figure(figsize=(15,15))
col1 = ['Frequency','Recency','Profit']
for i in range(3):
    plt.subplot(2,2,i+1)
    sns.boxplot(x=RFM_final['Category'],y=RFM_final[col1[i]])
    plt.xlabel('Category',size=12)
    plt.ylabel(f'{col1[i]}',size=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


# ##### Observations:
# Top 1000 customers have properties - 
# 1. Average frequency of purchase in a year is 8 times (compared to 4)
# 2. Have a recency of 30 days i.e. their last visit was 30 days ago (compared to 78 days)
# 3. Have a average profit of Rs.4884 (compared to Rs.2267)

# In[420]:


# Relationship between RFM

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
plt.scatter(x=RFM_final.Recency,y=RFM_final.Profit,color='g')    
plt.xlabel('Recency(in days)',size=12)
plt.ylabel('Profit(in rupees)',size=12)
plt.axvline(x=30,color='black', linestyle='--',lw=1.5)
plt.axhline(y=4884,color='black',linestyle='--',lw=1.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2,2,2)
plt.scatter(x=RFM_final.Frequency,y=RFM_final.Profit,color='g')    
plt.xlabel('Frequency',size=12)
plt.ylabel('Profit(in rupees)',size=12)
plt.axvline(x=8,color='black', linestyle='--',lw=1.5)
plt.axhline(y=4884,color='black',linestyle='--',lw=1.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2,2,3)
plt.scatter(x=RFM_final.Recency,y=RFM_final.Frequency,color='g')    
plt.xlabel('Recency(in days)',size=12)
plt.ylabel('Frequency',size=12)
plt.axvline(x=30,color='black', linestyle='--',lw=1.5)
plt.axhline(y=8,color='black',linestyle='--',lw=1.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()


# ##### Observations:
# 1. Profit vs Recency
#     - We see that as recency get higher the profit will get lower. 
#     - Cluster of data was within 100 days recency.
#     - The top left corner belongs to those customers whose recency was within 30 days and profit greater than 5000. 
# 2. Profit vs Frequency
#     - As the frequency get higher so the profit.
#     - The top right corner belongs to those customer whose frequency is more than 8 times and profit greater than 5000.
# 3. Recency vs Frequency
#     - As the recency get higher the frequency get lower.
#     - The top left corner belongs to those customers whose recency was within 30 days and frequency more than 8 times.

# In[389]:


# List of top 1000 customers

top_customers = RFM_final[RFM_final.Category==1]
top_customers.reset_index(inplace=True)
top_customers.drop('index',axis=1,inplace=True)
top_customers.head(10)


# In[414]:


# Saving data locally
# top_customers.to_excel('KPMG_top1000_customers_list.xlsx')


# #### Another Way

# In[391]:


RFM3 = RFM[['customer_id','Recency','Frequency','Profit']]
RFM3.head()


# In[392]:


RFM3.describe()


# In[393]:


# Cutting at 25%, 50% and 75% to divide the customers into 4 bins

RFM3['Recency_Score'] = pd.cut(RFM3.Recency, bins=[0,20,47,90,355],labels=[4,3,2,1])
RFM3['Frequency_Score'] = pd.cut(RFM3.Frequency, bins=[0,5,6,8,15],labels=[1,2,3,4])
RFM3['Profit_Score'] = pd.cut(RFM3.Profit, bins=[0,1775,2772,4073,11670],labels=[1,2,3,4])


# In[394]:


RFM3.head()


# In[395]:


RFM3['Recency_Score'] = RFM3['Recency_Score'].astype(int)
RFM3['Frequency_Score'] = RFM3['Frequency_Score'].astype(int)
RFM3['Profit_Score'] = RFM3['Profit_Score'].astype(int)

RFM3['RFM_Score'] = 100*RFM3.Recency_Score + 10*RFM3.Frequency_Score + RFM3.Profit_Score
RFM3.head()


# In[396]:


RFM3.info()


# In[397]:


RFM3.describe(include='all')


# In[398]:


RFM3['Customer_title'] = pd.cut(RFM3.RFM_Score, bins=[0,212,312,412,445],labels=['Bronze','Silver','Gold','Platinum'])
RFM3.head()


# In[399]:


RFM3.describe(include='all')


# In[400]:


RFM3 = RFM3.merge(new_df,how='left',on='customer_id')
RFM3.head()


# In[193]:


# Saving the data locally
RFM3.to_excel('KPMG_online_sol_final_table.xlsx')


# In[ ]:




