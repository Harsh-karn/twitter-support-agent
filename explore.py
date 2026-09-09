import pandas as pd
df = pd.read_csv('data/processed/apple_support_pairs.csv')
sample = df['customer_text'].sample(40, random_state=42).tolist()
with open('sample_tweets.txt', 'w', encoding='utf-8') as f:
    for s in sample:
        f.write(s + '\n')
