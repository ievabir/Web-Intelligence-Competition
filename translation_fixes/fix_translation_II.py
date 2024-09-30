#Was only used to fix some issues with previous translation logic without 
#rerunning translation. No longer relevant.

import pandas as pd

df_1 = pd.read_csv('translation_mid.csv')
df_2 = pd.read_csv('translated_file.csv')

df_1 = df_1[['id','title_t','description_t']]

# Ensure correct data types and fill NaNs if necessary
df_2['description'] = df_2['description'].astype(str).fillna('')
df_1['description_t'] = df_1['description_t'].astype(str).fillna('')

# Merge the DataFrames on the 'id' column
df_combined = df_2.merge(df_1[['id', 'title_t', 'description_t']], on='id', how='left')

# Update the original columns with the corrected translations
df_combined['title'] = df_combined['title_t'].combine_first(df_combined['title'])
df_combined['description'] = df_combined['description_t'].combine_first(df_combined['description'])

# Drop the temporary columns used for merging
df_combined.drop(columns=['title_t', 'description_t'], inplace=True)

# Save the updated DataFrame
df_combined.to_csv('translated_file_corrected.csv', index=False)
