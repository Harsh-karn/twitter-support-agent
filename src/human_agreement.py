import pandas as pd

def check_human_agreement():
    print("Loading human agreement dataset (n=20)...")
    try:
        df = pd.read_csv("data/human_agreement.csv")
    except FileNotFoundError:
        print("Error: data/human_agreement.csv not found.")
        return
        
    total_samples = len(df)
    
    # Calculate how many are within a +/- 2 point margin (out of 20 points)
    df['within_margin'] = abs(df['human_total'] - df['llm_total']) <= 2
    agreed = df['within_margin'].sum()
    
    agreement_rate = (agreed / total_samples) * 100
    
    print("\n--- LLM-as-Judge vs Human Agreement ---")
    print(f"Total Evaluated: {total_samples}")
    print(f"Agreed within ±2 points: {agreed}")
    print(f"Agreement Rate: {agreement_rate:.1f}%")
    
    if agreement_rate >= 80:
        print("\nConclusion: The LLM Judge strongly agrees with human evaluation (>80%), making it a reliable proxy for reply quality.")
    else:
        print("\nConclusion: The LLM Judge does not strongly agree with human evaluation. Further calibration required.")
        
    print("\nDetailed Discrepancies (where difference > 2):")
    discrepancies = df[~df['within_margin']]
    if discrepancies.empty:
        print("None! All LLM scores were within ±2 points of the human score.")
    else:
        for idx, row in discrepancies.iterrows():
            print(f"- Reply: '{row['reply']}'")
            print(f"  Human Score: {row['human_total']} | LLM Score: {row['llm_total']}")
            print(f"  Difference: {abs(row['human_total'] - row['llm_total'])} points\n")

if __name__ == "__main__":
    check_human_agreement()
