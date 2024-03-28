import os
import pandas as pd

def sample(df, label, sampling_ratio):
    print(label)
    df_label = df[df['label'] == label]
    sampled_df = df_label.sample(frac=sampling_ratio, random_state=42)
    #sampled_df.drop(columns=[label], inplace=True)
    df = df[df['label'] != label]
    df_sampled = pd.concat([df, sampled_df], ignore_index=True)
    return df_sampled

curdir = os.getcwd()
df = pd.read_csv(os.path.join(curdir, 'datasets', 'df8.csv'))
filters = ['DDoS', 'Mirai', 'DoS', 'Spoofing', 'Benign']
df_new = df[df['label'].isin(filters)]
cnt = df_new['label'].value_counts()
print(cnt)

df_sampled = sample(df_new, 'DDoS', 0.03)
df_sampled = sample(df_sampled, 'DoS', 0.1)
df_sampled = sample(df_sampled, 'Mirai', 0.3)
df_sampled = sample(df_sampled, 'Benign', 0.7)

cnt = df_sampled['label'].value_counts()
print(cnt)
df_sampled.to_csv(os.path.join(curdir, 'datasets', 'filtered_df.csv'))