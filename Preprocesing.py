import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

argv = sys.argv
argc = len(argv)
if(argc < 5 or argc == 6 or argc > 8):
  print("Usage: python Preprocesing.py --dataset <directory> --save_directory <directory> [--merge [name][description]]")
  sys.exit(-1)
  
dataset_directory = argv[2]
save_directory = argv[4]

df = pd.read_csv(dataset_directory, sep='\t')

if(argc == 7 and argv[argc-1] == "name"):
  df['full_text'] = df['name'].where(pd.notna(df['name']), '') + ' ' + df['tweet_text']
elif(argc == 7 and argv[argc-1] == "description"):
  df['full_text'] = df['description'].where(pd.notna(df['description']), '') + ' ' + df['tweet_text']
elif(argc == 8):
  df['full_text'] = df['name'].where(pd.notna(df['name']), '') + ' ' + df['description'].where(pd.notna(df['description']), '') + ' ' + df['tweet_text']
else:
  df['full_text'] = df['tweet_text']
  
df = df[['full_text', 'class']]
df = df.rename(columns={'class':'sentiment'})
train, test = train_test_split(df, test_size=0.3)
test = test[['full_text']]

output_file = os.path.join(save_directory, 'train.tsv')
train.to_csv(output_file, sep='\t')
output_file = os.path.join(save_directory, 'test.tsv')
test.to_csv(output_file, sep='\t')
