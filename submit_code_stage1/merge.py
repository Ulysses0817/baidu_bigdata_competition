import numpy as np
import pickle
import pandas as pd

with open("../data/stack.pkl", 'rb') as f:
    model1 = pickle.load(f)
with open("../data/stack2.pkl", 'rb') as f:
    model2 = pickle.load(f)

result = 0
for key in [str(i) for i in range(5)]:
    result += (model1[key] + model2[key])/2
result /= 5
result = result.argmax(axis=1)

sample_submission_df = pd.read_csv("../data/test.csv")
del sample_submission_df['Target']
sample_submission_df.columns = ['AreaID']
sample_submission_df['AreaID'] = sample_submission_df['AreaID'].apply(lambda x: str(x).zfill(6))
sample_submission_df['CategoryID'] = [str(x+1).zfill(3) for x in result]
sample_submission_df.sort_values(by='AreaID', inplace=True)
sample_submission_df[['AreaID','CategoryID']].to_csv('./submit/merge.csv', sep="\t", header=False, index=None)