#!/usr/bin/env python3
"""
Dataset transformation script for rich email datasets.
Converts datasets with multiple columns (sender, receiver, subject, body, etc.) 
to the format expected by the phishing detection system.
"""
import pandas as pd
import argparse
import os

def transform_dataset(input_file: str, output_file: str = None):
    """
    Transform a rich email dataset to the expected format.
    
    Expected input columns: sender, receiver, date, subject, body, label, urls
    Output columns: text, label
    """
    # Read the dataset
    print(f"📂 Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Verify required columns
    required_cols = ['sender', 'receiver', 'date', 'subject', 'body', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"✅ Found {len(df)} emails in dataset")
    print(f"📊 Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Transform data
    transformed_data = []
    
    for _, row in df.iterrows():
        # Combine all text features into a single field
        text_parts = []
        
        # Add subject (usually most important)
        if pd.notna(row['subject']) and str(row['subject']).strip():
            text_parts.append(f"Subject: {row['subject']}")
        
        # Add body content
        if pd.notna(row['body']) and str(row['body']).strip():
            text_parts.append(f"Body: {row['body']}")
        
        # Add sender information
        if pd.notna(row['sender']) and str(row['sender']).strip():
            text_parts.append(f"From: {row['sender']}")
        
        # Add URLs if present
        if 'urls' in df.columns and pd.notna(row['urls']) and str(row['urls']).strip():
            text_parts.append(f"URLs: {row['urls']}")
        
        # Combine all parts
        combined_text = " | ".join(text_parts)
        
        # Clean up the label
        label = str(row['label']).lower().strip()
        if label in ['1', 'true', 'phishing', 'spam', 'phish']:
            label = 'phish'
        elif label in ['0', 'false', 'legitimate', 'ham', 'normal']:
            label = 'ham'
        
        transformed_data.append({
            'text': combined_text,
            'label': label
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(transformed_data)
    
    # Remove any empty text entries
    output_df = output_df[output_df['text'].str.strip() != '']
    
    print(f"✅ Transformed {len(output_df)} valid emails")
    print(f"📊 Final label distribution: {output_df['label'].value_counts().to_dict()}")
    
    # Save transformed dataset
    if output_file is None:
        output_file = input_file.replace('.csv', '_transformed.csv')
    
    output_df.to_csv(output_file, index=False)
    print(f"💾 Saved transformed dataset to {output_file}")
    
    # Show sample
    print("\n📋 Sample transformed entries:")
    for i in range(min(3, len(output_df))):
        print(f"\nEntry {i+1}:")
        print(f"Label: {output_df.iloc[i]['label']}")
        print(f"Text: {output_df.iloc[i]['text'][:200]}...")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Transform rich email dataset for phishing detection')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: File {args.input_file} not found!")
        return
    
    success = transform_dataset(args.input_file, args.output)
    
    if success:
        output_file = args.output or args.input_file.replace('.csv', '_transformed.csv')
        print(f"\n🚀 Ready to upload! Use this command:")
        print(f"curl -X POST 'http://localhost:8000/api/admin/datasets' \\")
        print(f"  -F 'file=@{output_file}' \\")
        print(f"  -F 'name=Phishing Email Dataset' \\")
        print(f"  -F 'notes=Transformed dataset with sender, subject, body, and URL features'")

if __name__ == "__main__":
    main()