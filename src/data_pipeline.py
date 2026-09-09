import pandas as pd
import re
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Normalize extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data(input_csv, output_csv, brand="AppleSupport", sample_size=2000):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Total rows: {len(df)}")
    
    # 1. Get all brand replies
    brand_replies = df[(df['author_id'] == brand) & (df['inbound'] == False)].copy()
    print(f"Total {brand} replies: {len(brand_replies)}")
    
    # 2. Get the tweets they are responding to (customer tweets)
    # in_response_to_tweet_id in the brand reply points to the customer's tweet_id
    brand_replies = brand_replies.dropna(subset=['in_response_to_tweet_id'])
    
    customer_tweets = df[df['tweet_id'].isin(brand_replies['in_response_to_tweet_id'])].copy()
    
    # Merge to create pairs
    # Rename columns to distinguish
    customer_tweets = customer_tweets[['tweet_id', 'text']].rename(columns={'tweet_id': 'customer_tweet_id', 'text': 'customer_text'})
    brand_replies = brand_replies[['in_response_to_tweet_id', 'text']].rename(columns={'in_response_to_tweet_id': 'customer_tweet_id', 'text': 'brand_text'})
    
    pairs = pd.merge(customer_tweets, brand_replies, on='customer_tweet_id')
    
    print(f"Reconstructed pairs: {len(pairs)}")
    
    # 3. Clean text
    print("Cleaning text...")
    pairs['customer_text'] = pairs['customer_text'].apply(clean_text)
    pairs['brand_text'] = pairs['brand_text'].apply(clean_text)
    
    # Remove empty pairs
    pairs = pairs[(pairs['customer_text'] != '') & (pairs['brand_text'] != '')]
    
    # 4. Subsample
    if len(pairs) > sample_size:
        print(f"Subsampling to {sample_size} pairs...")
        sampled_pairs = pairs.sample(n=sample_size, random_state=42)
    else:
        sampled_pairs = pairs
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sampled_pairs.to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")

if __name__ == "__main__":
    input_path = "data/raw/twcs/twcs.csv"
    output_path = "data/processed/apple_support_pairs.csv"
    
    if not os.path.exists(input_path):
        # Maybe it's directly in data/raw
        input_path = "data/raw/twcs.csv"
        
    process_data(input_path, output_path, brand="AppleSupport", sample_size=2000)
