#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()  # Start the timer


# In[2]:


import requests
import pandas as pd

# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Sales Invoice'

url = base_url + endpoint

# Define the parameters for the request
params = {
    'fields': '["name","customer","base_grand_total","due_date","posting_date" ,"outstanding_amount","credit_rating_not_to_consider"]',
    'limit_start': 0,  # Start from the first record
    'limit_page_length': 1000,  # Request a large number of records per page
    'filters': '[["company", "=", "Electrolab India Pvt. Ltd."]]'
}

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

# Initialize variables for pagination
limit_start = 0
limit_page_length = 1000
all_data = []

# Loop to handle pagination
while True:
    # Update limit_start in params for each iteration
    params['limit_start'] = limit_start
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Process the response
        data = response.json()
        if 'data' in data:
            current_page_data = data['data']
            all_data.extend(current_page_data)
            
            # Check if there are more records
            if len(current_page_data) < limit_page_length:
                break  # No more records, exit loop
            else:
                limit_start += limit_page_length  # Move to the next page
        else:
            break  # Exit if no data key in response
            
    except requests.exceptions.RequestException as e:
        print(f"Error Occured. Retrying in 5 seconds...")
        time.sleep(5) 

# Create DataFrame
invoice = pd.json_normalize(all_data)
invoice.rename(columns={'base_grand_total': 'grand_total'}, inplace=True)
# Drop rows where custom_credit_rate is 1
invoice = invoice[invoice["credit_rating_not_to_consider"] != 1]
# Display the first few rows of the DataFrame
invoice.head()


# In[3]:




# In[4]:


import requests
import pandas as pd

# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer'
url = base_url + endpoint

# Define the headers for the request
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

# Create a session with retries
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Initialize variables for pagination
limit_start = 0
limit_page_length = 1000
all_data = []

# Loop to handle pagination
while True:
    # Define the parameters for the request
    params = {
        'fields': '["customer_name","customer_group","bad_debts","custom_credit_rate"]',
        'limit_start': limit_start,
        'limit_page_length': limit_page_length
    }
    
    try:
        response = session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Process the response
        data = response.json()
        if 'data' in data:
            customer_data = data['data']
            if not customer_data:
                break  # No more data to fetch
            all_data.extend(customer_data)
            limit_start += limit_page_length
        else:
            break  # Exit if no data key in response
            
    except requests.exceptions.RequestException as e:
        print(f"Error Occured. Retrying in 5 seconds...")
        time.sleep(5) 

# Convert the collected data to a DataFrame
customer = pd.json_normalize(all_data)

# Display the first few rows of the DataFrame
customer.head()


# In[5]:


# Filter out 'Foreign Customers' and 'All Customer Groups'
customer_group = customer[~customer['customer_group'].isin(['Foreign Customers', 'All Customer Groups'])]


# In[6]:


customer_group.rename(columns = {'customer_name':'customer'},inplace = True)
customer_group.head()


# In[7]:


# Merge the filtered customer DataFrame with out_df on 'customer'
invoice_df = pd.merge(invoice, customer_group, on='customer', how='left')

# Replace 'customer' with 'customer_group' where available
invoice_df['customer'] = invoice_df['customer_group'].combine_first(invoice_df['customer'])

# Drop the now redundant 'customer_group' column
invoice_df.drop(columns=['customer_group'], inplace=True)
invoice_df.head()


# In[8]:


from datetime import datetime


# In[9]:


# Assuming 'invoice' is your DataFrame
# Filtering rows with 'grand_total' greater than or equal to zero
invoice_df = invoice_df[invoice_df['grand_total'] > 0]




# In[10]:


# Assuming 'invoice' is your DataFrame
invoice_df['due_date'] = pd.to_datetime(invoice_df['due_date'])
invoice_df['current_date'] = pd.to_datetime(datetime.now().date())
invoice_df['days_until_due'] = (invoice_df['current_date'] - invoice_df['due_date']).dt.days

invoice_df.head()


# In[11]:


# Filter out rows with negative values
out_df = invoice_df[(invoice_df['outstanding_amount'] > 1) & (invoice_df['days_until_due'] > 0)]
out_df.head()


# In[12]:


out_df['outstanding_score'] = (out_df['outstanding_amount'] * out_df['days_until_due'])/100


# In[13]:


out_df = out_df[['customer','outstanding_amount','days_until_due','outstanding_score']]


# In[14]:


# sum of 'Outstanding Score', and mean of 'Age (Days)'
grouped_data = out_df.groupby('customer').agg({
    'outstanding_amount': 'sum',
    'outstanding_score': 'sum',
    'days_until_due': 'mean'
}).reset_index()
grouped_data.head()


# In[15]:


# Calculate the required statistics
min_score = grouped_data['outstanding_score'].min()
max_score = grouped_data['outstanding_score'].max()
median_score = grouped_data['outstanding_score'].median()
percentile_75_score = grouped_data['outstanding_score'].quantile(0.75)
percentile_25_score = grouped_data['outstanding_score'].quantile(0.25)


# Print the results
print(f"Minimum Outstanding Score: {min_score}")
print(f"Maximum Outstanding Score: {max_score}")
print(f"Median Outstanding Score (50th Percentile): {median_score}")
print(f"75th Percentile Outstanding Score: {percentile_75_score}")
print(f"25th Percentile Outstanding Score: {percentile_25_score}")


# In[16]:


def assign_credit_rate(outstanding_score_sum):
    total_score = outstanding_score_sum 
    

    if total_score <= 23675.52:
        return 0
    elif total_score <= 155126.40:
        return 1
    elif total_score <= 300000:
        return 2
    elif total_score <= 500000:
        return 3
    elif total_score <= 700000:
        return 4
    elif total_score <= 900000:
        return 5
    elif total_score <= 1000000:
        return 6
    else:
        return 7

# Example usage:
grouped_data['outstanding_rate'] = grouped_data.apply(lambda row: assign_credit_rate(row['outstanding_score']), axis=1)
grouped_data.head()


# In[17]:


# Sorting the DataFrame by outstanding amount in descending order
sorted_df = grouped_data.sort_values(by='outstanding_rate', ascending=False)



# In[18]:


import requests
import pandas as pd

# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Payment Entry'
url = base_url + endpoint

# Define the headers for the request
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

# Create a session with retries
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Initialize variables for pagination
limit_start = 0
limit_page_length = 1000
all_data = []

# Loop to handle pagination
while True:
    # Define the parameters for the request
    params = {
        'fields': '["name","references.reference_name","posting_date","references.total_amount","references.due_date","references.outstanding_amount","paid_amount"]',
        'limit_start': limit_start,
        'limit_page_length': limit_page_length,
        'filters': '[["payment_type", "=", "Receive"], ["party_type", "=", "Customer"],["is_advance", "=", "No"]]'
    }
    
    try:
        response = session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Process the response
        data = response.json()
        if 'data' in data:
            payment_data = data['data']
            if not payment_data:
                break  # No more data to fetch
            all_data.extend(payment_data)
            limit_start += limit_page_length
        else:
            break  # Exit if no data key in response
            
    except requests.exceptions.RequestException as e:
        print(f"Error Occured. Retrying in 5 seconds...")
        time.sleep(5) 

# Convert the collected data to a DataFrame
payment = pd.json_normalize(all_data)



# In[19]:


import requests
import pandas as pd


# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Journal Entry'

params = {
    'fields': '["voucher_type", "accounts.is_advance", "posting_date","accounts.credit","accounts.reference_name","accounts.reference_type"]',  # Include required fields
    'limit_start': 0,
    'limit_page_length': 1000,
    'filters': '[["voucher_type", "=", "Journal Entry"], ["total_credit", ">", "0"]]'
}

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

all_data = []



# Make the GET request
while True:
    response = requests.get(base_url + endpoint, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        current_page_data = data.get('data', [])  # Data from the current page
        all_data.extend(current_page_data)  # Append data from the current page
        
        # Check if there are more records
        if len(current_page_data) < params['limit_page_length']:
            break  # No more records, exit loop
        else:
            params['limit_start'] += params['limit_page_length']  # Move to the next page
    else:
        print('Request failed with status code:', response.status_code)
        time.sleep(5)

# Create DataFrame
journal = pd.DataFrame(all_data)
# Filter out rows where reference_type is not "Sales Invoice"
journal = journal[(journal['reference_type'] == 'Sales Invoice') & (journal['is_advance'] == 'No')]

journal.head()




# In[20]:


# Drop rows with missing values in 'reference_name' column
journal.dropna(subset=['reference_name'], inplace=True)

# Convert 'posting_date' column to datetime
journal['posting_date'] = pd.to_datetime(journal['posting_date'])

# Sort DataFrame by 'posting_date' from latest to oldest
journal = journal.sort_values(by='posting_date', ascending=False)
# Filter the DataFrame based on the condition
#paid = paid[paid['reference_name'].str.startswith(('1920''2221','2223','2324','2425'))]

# Display the first few rows of the filtered DataFrame
journal.head()


# In[21]:


# sum of 'Outstanding Score', and mean of 'Age (Days)'
journal_df = journal.groupby('reference_name').agg({
    'posting_date': 'first',
    'credit':'sum'
}).reset_index()
journal_df.head()


# In[22]:


journal_df.rename(columns = {'reference_name':'name',
                         'posting_date':'payment_date',
                            'credit':'paid_amount'},inplace = True)
journal_df.head()


# In[23]:


# Drop rows with missing values in 'reference_name' column
payment.dropna(subset=['reference_name'], inplace=True)

# Convert 'posting_date' column to datetime
payment['posting_date'] = pd.to_datetime(payment['posting_date'])

# Sort DataFrame by 'posting_date' from latest to oldest
payment = payment.sort_values(by='posting_date', ascending=False)
# Filter the DataFrame based on the condition
#paid = paid[paid['reference_name'].str.startswith(('1920''2221','2223','2324','2425'))]

# Display the first few rows of the filtered DataFrame
payment.head()


# In[24]:


# sum of 'Outstanding Score', and mean of 'Age (Days)'
payment_df = payment.groupby('reference_name').agg({
    'posting_date': 'first',
    'paid_amount':'sum'
}).reset_index()
payment_df.head()


# In[25]:


payment_df.rename(columns = {'reference_name':'name',
                         'posting_date':'payment_date'},inplace = True)


# In[26]:


# Concatenate the two DataFrames together
paid_df = pd.concat([payment_df, journal_df], ignore_index=True)
paid_df.head()


# In[27]:


# Drop duplicates based on the 'name' column and keep the last occurrence
paid_df = paid_df.drop_duplicates(subset='name', keep='last')


# In[28]:


invoice_df.head()
paid_df1 = invoice_df[invoice_df['outstanding_amount'] == 0][['name', 'customer', 'posting_date', 'grand_total', 'outstanding_amount']]
paid_df1.head()


# In[29]:


paid_final = pd.merge(paid_df1, paid_df, on='name', how='left')
paid_final.head()


# In[30]:


# Drop rows where 'paid_amount' is NaN
paid_final = paid_final.dropna(subset=['paid_amount'])
paid_final.head()


# In[31]:


# Convert 'posting_date' and 'payment_date' to datetime objects
paid_final['posting_date'] = pd.to_datetime(paid_final['posting_date'])
paid_final['payment_date'] = pd.to_datetime(paid_final['payment_date'])

# Calculate the difference between 'payment_date' and 'posting_date'
paid_final['date_difference'] = paid_final['payment_date'] - paid_final['posting_date']


# In[32]:


#paid_final['on_time'] = (paid_final['payment_date'] - paid_final['posting_date']).dt.days <= 30

# Filter rows where outstanding_amount is zero
filtered_paid_final = paid_final[paid_final['outstanding_amount'] == 0]
# Calculate on-time payment (within 30 days)
filtered_paid_final.head()


# In[33]:


# Group by customer and calculate total payments and on-time payments
grouped_payments = filtered_paid_final.groupby('customer').agg(
    total_payments=('date_difference', 'mean'),
    total_invoice=('name', 'count')
).reset_index()
grouped_payments.head()


# In[34]:


# Convert total_payments to number of days
grouped_payments['on_time_score'] = grouped_payments['total_payments'].dt.total_seconds() / (24 * 3600)


# In[35]:


grouped_payments.head()


# In[36]:


# Calculate average payment period per invoice
#grouped_payments['on_time_score'] = grouped_payments['total_payments_days'] / grouped_payments['total_invoice']
#grouped_payments.head()


# In[37]:


# Calculate the required statistics
min_score = grouped_payments['on_time_score'].min()
max_score = grouped_payments['on_time_score'].max()
median_score = grouped_payments['on_time_score'].median()
percentile_75_score = grouped_payments['on_time_score'].quantile(0.75)
percentile_25_score = grouped_payments['on_time_score'].quantile(0.10)


# Print the results
print(f"Minimum Outstanding Score: {min_score}")
print(f"Maximum Outstanding Score: {max_score}")
print(f"Median Outstanding Score (50th Percentile): {median_score}")
print(f"75th Percentile Outstanding Score: {percentile_75_score}")
print(f"25th Percentile Outstanding Score: {percentile_25_score}")


# In[38]:


# Function to determine credit_rate based on total_payments_days
def calculate_credit_rate(days):
    if days <= 0:
        return 0
    elif days <= 45:
        return 1
    elif days <= 65:
        return 2
    else:
        return 3

# Apply function to calculate credit_rate and update the on_time_score column
grouped_payments['credit_rate'] = grouped_payments['on_time_score'].apply(calculate_credit_rate)

# Replace on_time_score with credit_rate
grouped_payments['on_time_score'] = grouped_payments['credit_rate']

# Drop the temporary credit_rate column as it's now merged into on_time_score
grouped_payments = grouped_payments.drop(columns=['credit_rate'])
grouped_payments = grouped_payments[['customer','on_time_score']]
grouped_payments.head()


# In[39]:


#grouped_payments['on_time_score'] = (3 - ((grouped_payments['on_time_payments']/grouped_payments['total_payments'] )) * 3).round(1)
#grouped_payments = grouped_payments[['customer','on_time_score']]
#grouped_payments.head()


# In[40]:


# Group by customer and sum the outstanding and grand total amount
grouped_df = invoice_df.groupby('customer').agg({
    'grand_total': 'sum',
    'outstanding_amount': 'sum'
}).reset_index()


# In[41]:


grouped_df.head()


# In[42]:


import numpy as np
# Calculate percentiles for grand total
percentiles = np.percentile(grouped_df['grand_total'], [88, 93, 95,98])

# Function to assign scores based on percentiles
def assign_score(grand_total):
    if grand_total >= percentiles[3]:
        return 4
    elif grand_total >= percentiles[2]:
        return 3
    elif grand_total >= percentiles[1]:
        return 2
    elif grand_total >= percentiles[0]:
        return 1
    else:
        return 0

# Apply the scoring function to each customer
grouped_df['paid_score'] = grouped_df['grand_total'].apply(assign_score)
grouped_df.head()


# In[43]:




# In[44]:


# Calculate the required statistics

max_score = grouped_df['grand_total'].max()
median_score = grouped_df['grand_total'].median()

percentile_88_score = grouped_df['grand_total'].quantile(0.88)
percentile_93_score = grouped_df['grand_total'].quantile(0.93)
percentile_95_score = grouped_df['grand_total'].quantile(0.95)
percentile_98_score = grouped_df['grand_total'].quantile(0.98)


# Print the results

print(f"Maximum Grand Total: {max_score}")
print(f"Median Grand Total (50th Percentile): {median_score}")
print(f"88th Percentile Grand Total: {percentile_88_score}")
print(f"93th Percentile Grand Total: {percentile_93_score}")
print(f"95th Percentile Grand Total: {percentile_95_score}")
print(f"98th Percentile Grand Total: {percentile_98_score}")


# In[45]:




# In[46]:


grouped_data = grouped_data[['customer','outstanding_rate']]
grouped_data.head()


# In[47]:


# Merging the paid_score and outstanding_rate DataFrames on 'customer' column
merged_df = pd.merge(grouped_df, grouped_data, on='customer', how='outer')

# Merging the result with on_time_score DataFrame on 'customer' column
merged_df = pd.merge(merged_df, grouped_payments, on='customer', how='outer')
merged_df.head()


# In[48]:


merged_df.fillna(0, inplace=True)
merged_df.head()


# In[49]:


merged_df['credit_rate'] = merged_df['on_time_score'] + merged_df['outstanding_rate'] - merged_df['paid_score']


# In[50]:


# Make credit_rate zero if negative
merged_df['credit_rate'] = merged_df['credit_rate'].apply(lambda x: 0 if x < 0 else x)
merged_df.head()


# In[51]:




# In[52]:


# Find maximum credit rate
max_credit_rate = merged_df['credit_rate'].max()

print("Maximum Credit Rate:", max_credit_rate)


# In[53]:


import requests
import pandas as pd

# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer'

# Define the parameters for the request
params = {
    'fields': '["name","customer_name","customer_group","bad_debts","custom_credit_rate"]',
    'limit_start': 0,  # Start from the first record
    'limit_page_length': 1000,  # Request a large number of records per page

}

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

all_data = []

# Make the GET request
while True:
    response = requests.get(base_url + endpoint, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        current_page_data = data['data']  # Data from the current page
        all_data.extend(current_page_data)  # Append data from the current page
        
        # Check if there are more records
        if len(current_page_data) < params['limit_page_length']:
            break  # No more records, exit loop
        else:
            params['limit_start'] += params['limit_page_length']  # Move to the next page
    else:
        print(f"Error Occured. Retrying in 5 seconds...", response.status_code)
        time.sleep(5) 

# Create DataFrame
customer = pd.DataFrame(all_data)
customer.head()


# In[54]:


# Assuming you have a DataFrame named df containing the data
customer['custom_credit_rate'] = customer['custom_credit_rate'].astype(float)
customer.head()


# In[55]:


merged_df.rename(columns = {'customer':'name','outstanding_rate':'ouststanding_rating','on_time_score':'past__payment_history'},inplace = True)
merged_df = merged_df[['name','credit_rate','past__payment_history','ouststanding_rating','paid_score']]
merged_df.head()
group = merged_df.copy()
group.head()


# In[56]:


group.rename(columns = {'name':'customer_group'}, inplace = True)
group.head()


# In[57]:


# Filter out 'Foreign Customers' and 'All Customer Groups'
filtered_final = customer[~customer['customer_group'].isin(['Foreign Customers', 'All Customer Groups'])]


individual_final = customer[customer['customer_group'].isin(['Foreign Customers', 'All Customer Groups'])]


# In[58]:


filtered_final.head()


# In[59]:


customer_group = pd.merge(group,filtered_final, on = 'customer_group', how = 'inner')
customer_group.head()


# In[60]:


customer = pd.merge(merged_df,individual_final, on = 'name', how = 'inner')
customer.head()


# In[61]:


customer_group.shape


# In[62]:


customer.shape


# In[63]:


# Concatenate DataFrames vertically
df_combined = pd.concat([customer_group, customer], ignore_index=True)


# In[64]:


# Convert 'bad_debts' to binary values (1 for 'Yes', 0 for 'None'), handling missing values
df_combined['bad_debts'] = df_combined['bad_debts'].apply(lambda x: 1 if pd.notna(x) and x.lower() == 'yes' else 0)

# Update 'credit_rate' based on 'bad_debts'
df_combined['credit_rate'] += df_combined['bad_debts']


# In[65]:


#def round_value(x):
    #decimal_part = x - int(x)  # Get the decimal part of the number
    #if decimal_part >= 0.7:
        #return int(x) + 1  # Round up
    #elif decimal_part <= 0.3:
        #return int(x) + 0.5  # Round to the nearest 0.5
    #else:
        #return round(x * 2) / 2  # Otherwise, apply the default rounding

# Apply the rounding function to each column
#df_combined['credit_rate'] = df_combined['credit_rate'].apply(round_value)
#df_combined.head()


# In[66]:




# In[67]:


import requests
import pandas as pd

# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer'

# Define the parameters for the request
params = {
    'fields': '["name","customer_name","customer_group","custom_past_payment_entry","custom_outstanding_rating","custom_credit_rate","custom_business_quantitate_score"]',
    'limit_start': 0,  # Start from the first record
    'limit_page_length': 1000,  # Request a large number of records per page

}

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

all_data = []

# Make the GET request
while True:
    response = requests.get(base_url + endpoint, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        current_page_data = data['data']  # Data from the current page
        all_data.extend(current_page_data)  # Append data from the current page
        
        # Check if there are more records
        if len(current_page_data) < params['limit_page_length']:
            break  # No more records, exit loop
        else:
            params['limit_start'] += params['limit_page_length']  # Move to the next page
    else:
        print(f"Error Occured. Retrying in 5 seconds...", response.status_code)
        time.sleep(5) 

# Create DataFrame
check_df = pd.DataFrame(all_data)
check_df.head()


# In[68]:


concatenated_df = df_combined[['name','customer_group','bad_debts','credit_rate','past__payment_history','ouststanding_rating','paid_score']]
concatenated_df.rename(columns = {'credit_rate':'custom_credit_rate','past__payment_history':'custom_past_payment_entry','ouststanding_rating':'custom_outstanding_rating','paid_score':'custom_business_quantitate_score'},inplace = True)
concatenated_df.head()


# In[69]:


# Convert custom_credit_rate column in check_df to float
check_df['custom_credit_rate'] = check_df['custom_credit_rate'].astype(float)

# Merge the dataframes on 'name'
merged_df = pd.merge(check_df, concatenated_df, on='name', suffixes=('_check', '_concat'))

# Define condition for comparison
condition = ((merged_df['custom_credit_rate_check'] != merged_df['custom_credit_rate_concat']) |
             (merged_df['custom_past_payment_entry_check'] != merged_df['custom_past_payment_entry_concat']) |
             (merged_df['custom_outstanding_rating_check'] != merged_df['custom_outstanding_rating_concat']) |
           (merged_df['custom_business_quantitate_score_check'] != merged_df['custom_business_quantitate_score_concat']) )

# Filter rows based on condition
filtered_df = merged_df[condition]


# In[70]:


filtered_df.head()


# In[71]:


filtered_df.shape


# In[72]:


concatenated_df.shape


# In[73]:


import requests
import json

# Assuming you have pandas imported as pd and final_df is your dataframe

# Define the base URL and endpoint for customer details
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer'

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807',
    'Content-Type': 'application/json'
}

# Iterate through each row in the final_df dataframe
for index, row in filtered_df.iterrows():
    # Check if customer_name is nan, if so, skip this iteration
    if pd.isnull(row['name']):
        print("Skipping row with NaN customer name")
        continue
    
    # Extract necessary information
    customer_name = row['name']
    custom_credit_rate = row['custom_credit_rate_concat']
    custom_past_payment_entry = row['custom_past_payment_entry_concat']
    custom_outstanding_rating = row['custom_outstanding_rating_concat']
    custom_business_quantitate_score = row['custom_business_quantitate_score_concat']
    
    # Construct the URL for the specific customer
    url = f"{base_url}{endpoint}/{customer_name}"
    
    # Define the payload (body) for the PUT request
    payload = {
        "custom_credit_rate": custom_credit_rate,
        "custom_outstanding_rating": custom_outstanding_rating,
        "custom_past_payment_entry": custom_past_payment_entry,
        "custom_business_quantitate_score":custom_business_quantitate_score
    }
            
    # Convert payload to JSON format
    json_payload = json.dumps(payload)
            
    # Send PUT request
    response = requests.put(url, headers=headers, data=json_payload)
            
    # Check if request was successful
    if response.status_code == 200:
        print(f"Successfully updated data for {customer_name}")
    else:
        print(f"Failed to update data for {customer_name}. Status code: {response.status_code}")


# In[74]:


import requests
import json

# Assuming you have pandas imported as pd and final_df is your dataframe

# Define the base URL and endpoint for customer details
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer'

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807',
    'Content-Type': 'application/json'
}

# Iterate through each row in the final_df dataframe
for index, row in filtered_df.iterrows():
    # Check if customer_name is nan, if so, skip this iteration
    if pd.isnull(row['name']):
        print("Skipping row with NaN customer name")
        continue
    
    # Extract necessary information
    customer_name = row['name']
    custom_credit_rate = row['custom_credit_rate_concat']
    custom_past_payment_entry = row['custom_past_payment_entry_concat']
    custom_outstanding_rating = row['custom_outstanding_rating_concat']
    custom_business_quantitate_score = row['custom_business_quantitate_score_concat']
    
    # Construct the URL for the specific customer
    url = f"{base_url}{endpoint}/{customer_name}"
    
    # Define the payload (body) for the PUT request
    payload = {
        "custom_credit_rate": custom_credit_rate,
        "custom_outstanding_rating": custom_outstanding_rating,
        "custom_past_payment_entry": custom_past_payment_entry,
        "custom_business_quantitate_score":custom_business_quantitate_score
    }
            
    # Convert payload to JSON format
    json_payload = json.dumps(payload)
            
    # Send PUT request
    response = requests.put(url, headers=headers, data=json_payload)
            
    # Check if request was successful
    if response.status_code == 200:
        print(f"Successfully updated data for {customer_name}")
    else:
        print(f"Failed to update data for {customer_name}. Status code: {response.status_code}")


# In[75]:


import requests
import pandas as pd

# Define the base URL and endpoint for stock entry
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer Group'

# Define the parameters for the request
params = {
    'fields': '["name","credit_rate"]',
    'limit_start': 0,  # Start from the first record
    'limit_page_length': 1000,  # Request a large number of records per page

}

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807'
}

all_data = []

# Make the GET request
while True:
    response = requests.get(base_url + endpoint, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        current_page_data = data['data']  # Data from the current page
        all_data.extend(current_page_data)  # Append data from the current page
        
        # Check if there are more records
        if len(current_page_data) < params['limit_page_length']:
            break  # No more records, exit loop
        else:
            params['limit_start'] += params['limit_page_length']  # Move to the next page
    else:
        print(f"Error Occured. Retrying in 5 seconds...", response.status_code)
        time.sleep(5) 

# Create DataFrame
customer_group = pd.DataFrame(all_data)

customer_group.rename(columns = {'name':'customer_group','credit_rate':'erp_rates'},inplace = True)

customer_group.head()


# In[76]:


# Assuming you have a DataFrame named df containing the data
customer_group['erp_rates'] = customer_group['erp_rates'].astype(float)


# In[77]:


group_df = pd.merge(customer_group,group,on = 'customer_group', how = 'left')
group_df.fillna(0,inplace = True)


# In[78]:


group_df.head()


# In[79]:


# Check if customer_credit_rate is equal to credit_rate
equal_credit_rate = group_df['erp_rates'] == group_df['credit_rate']

# Drop rows where customer_credit_rate is equal to credit_rate
group_dff = group_df[~equal_credit_rate]


# In[80]:


group_dff.head()


# In[81]:


import requests
import json

# Assuming you have pandas imported as pd and final_df is your dataframe

# Define the base URL and endpoint for customer details
base_url = 'https://erpv14.electrolabgroup.com/'
endpoint = 'api/resource/Customer Group'

# Define the headers if needed
headers = {
    'Authorization': 'token 3ee8d03949516d0:6baa361266cf807',
    'Content-Type': 'application/json'
}

# Iterate through each row in the final_df dataframe
for index, row in group_dff.iterrows():
    # Check if customer_name is nan, if so, skip this iteration
    if pd.isnull(row['customer_group']):
        print("Skipping row with NaN customer name")
        continue
    
    # Extract necessary information
    customer_name = row['customer_group']
    credit_rate = row['credit_rate']
    past_payment_history = row['past__payment_history']
    outstanding_rating = row['ouststanding_rating']
    
    
    # Construct the URL for the specific customer
    url = f"{base_url}{endpoint}/{customer_name}"
    
    # Define the payload (body) for the PUT request
    payload = {
        "credit_rate": credit_rate,
        "outstanding_rating": outstanding_rating,
        "past_payment_history": past_payment_history
    }
            
    # Convert payload to JSON format
    json_payload = json.dumps(payload)
            
    # Send PUT request
    response = requests.put(url, headers=headers, data=json_payload)
            
    # Check if request was successful
    if response.status_code == 200:
        print(f"Successfully updated data for {customer_name}")
    else:
        print(f"Failed to update data for {customer_name}. Status code: {response.status_code}")


# In[82]:


from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Format the datetime object as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Print the formatted datetime
print('Updated at:', formatted_datetime)


# In[83]:


end_time = time.time()  # End the timer
execution_time = end_time - start_time  # Calculate execution time
execution_time = round(execution_time, 2)
print(f"Execution Time: {execution_time} seconds")


# In[ ]:





# In[ ]:




