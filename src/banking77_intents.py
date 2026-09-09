import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

def run_banking77_eval():
    print("Loading PolyAI/banking77 dataset...")
    try:
        dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dataset. Make sure you have internet access and HF datasets installed: {e}")
        return
    
    # Extract train and test splits
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    print(f"Loaded {len(train_df)} training examples and {len(test_df)} testing examples.")
    print(f"Number of unique intents: {train_df['label'].nunique()}")
    
    X_train, y_train = train_df['text'], train_df['label']
    X_test, y_test = test_df['text'], test_df['label']
    
    print("\nTraining TF-IDF + Logistic Regression Baseline...")
    start_time = time.time()
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    print("\nEvaluating on Test Set...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    
    print("\nEvaluation successfully completed without requiring any LLM API credits!")

if __name__ == "__main__":
    run_banking77_eval()
