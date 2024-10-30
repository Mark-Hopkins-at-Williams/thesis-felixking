import pandas as pd
import sys

def reorganize_csv(input_file, output_file):
    """
    Reorganize CSV data and cluster English pairs:
    1. English source pairs (eng_Latn -> X) at the top
    2. English target pairs (X -> eng_Latn) at the bottom
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_columns = ['model', 'source', 'target', 'bleu', 'chrf']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
        
        # Create a unique identifier for each language pair
        df['lang_pair'] = df['source'] + '-' + df['target']
        
        # Create new column names for each model's scores
        result = pd.DataFrame()
        result['source'] = df.groupby('lang_pair')['source'].first()
        result['target'] = df.groupby('lang_pair')['target'].first()
        
        # Add scores for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            result[f'bleu_{model}'] = model_data.set_index('lang_pair')['bleu']
            result[f'chrf_{model}'] = model_data.set_index('lang_pair')['chrf']
        
        # Reset index to make lang_pair a regular column
        result = result.reset_index(drop=True)
        
        # Split the dataframe into English source and English target
        eng_source = result[result['source'] == 'eng_Latn']
        eng_target = result[result['target'] == 'eng_Latn']
        other_pairs = result[
            (result['source'] != 'eng_Latn') & 
            (result['target'] != 'eng_Latn')
        ]
        
        # Sort each group alphabetically by target/source language
        eng_source = eng_source.sort_values('target')
        eng_target = eng_target.sort_values('source')
        
        # Concatenate the groups in desired order
        final_result = pd.concat([eng_source, other_pairs, eng_target])
        
        # Save to new CSV file
        final_result.to_csv(output_file, index=False)
        print(f"Successfully reorganized data and saved to {output_file}")
        
        # Print summary of clustering
        print(f"\nData summary:")
        print(f"English source pairs (eng_Latn → X): {len(eng_source)}")
        print(f"English target pairs (X → eng_Latn): {len(eng_target)}")
        print(f"Other language pairs: {len(other_pairs)}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.csv output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    reorganize_csv(input_file, output_file)