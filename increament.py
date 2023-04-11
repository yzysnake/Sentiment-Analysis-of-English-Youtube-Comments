import pandas as pd

# Read the CSV file
data = pd.read_csv('/Users/ziyuanye/Documents/PSU/2023 Spring/DS 340W/Final Project/sentiment.csv', delimiter=";")

# Filter rows where Label is 1 and Lan is 'EN'
filtered_data_1 = data[(data['label'] == 1) & (data['lan'] == 'EN')]

# Filter rows where Label is 2 and Lan is 'EN'
filtered_data_2 = data[(data['label'] == 2) & (data['lan'] == 'EN')]

# Select only 'Text' and 'Score' columns for both filters
result_1 = filtered_data_1[['text', 'score']]
result_2 = filtered_data_2[['text', 'score']]

# Revise the Score for the second filter, changing -2 to -1 and 2 to 1
result_2.loc[:, 'score'] = result_2['score'].replace({-2: -1, 2: 1})

# Combine both results
combined_result = pd.concat([result_1, result_2])

# Rename the columns
combined_result = combined_result.rename(columns={'text': 'comment', 'score': 'label'})

# Increment the 'label' column by 1 using .loc[]
combined_result.loc[:, 'label'] += 1

# Save the combined result to a new CSV file
combined_result.to_csv('Combined_Filtered_Sentiment.csv', index=False)
