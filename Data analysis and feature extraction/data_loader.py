import pandas as pd
import json
import os
from tqdm import tqdm
import timeit

DATA_FOLDER = "/home/indy-stg2/suresh2/cctweets_untaxonomy/general/"
#DATA_FOLDER = "/Users/alexzhu/Downloads/"
PATH_TWITTER_DATA_SRC = DATA_FOLDER+"tweet2021presentflattened.json"
SAMPLE = False
MAX_ROWS = 1_000

file_size = os.stat(PATH_TWITTER_DATA_SRC).st_size
print(f"File size is {file_size / 2**30:>3,.2f} GB")

nb_rows_read = 0
lines_json = []    

start = timeit.default_timer()
df = pd.DataFrame(index=None)

with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
    with open(PATH_TWITTER_DATA_SRC, "r") as f:
        for line in f:
            # limit the nb of iterations for test purposes
            if (SAMPLE == True) and (nb_rows_read >= MAX_ROWS):
                break
                    
            lines_json.append(json.loads(line))
            nb_rows_read += 1
            if nb_rows_read % 500_000 == 0:
                df_temp = pd.DataFrame(data=lines_json, index=None)
                df = pd.concat([df,df_temp])
                #del lines_json
                lines_json = []  
                print(f"{nb_rows_read:,} rows read")
                print('clear')
            if nb_rows_read == 4_000_000:
                df=df.drop(['entities','context_annotations','edit_history_tweet_ids','__twarc','attachments'],axis=1)
                df.to_csv('/home/indy-stg2/mlteam1/processed_data_1.csv.bz2', compression='bz2',index=False)
                #del df
                df = pd.DataFrame(index=None)


            read_bytes = len(line)
            if read_bytes:
                pbar.set_postfix(file=PATH_TWITTER_DATA_SRC[len(DATA_FOLDER):], refresh=False)
                pbar.update(read_bytes)
            
stop = timeit.default_timer()
time_diff = stop - start

print(f"==> total time to read data:        {time_diff/60:>10.0f} min. ({time_diff:.0f}s.)")
print(f"==> number of rows (= videos) read: {nb_rows_read:>10,}")

# create new dataframe with the json lines
df_temp = pd.DataFrame(data=lines_json, index=None)
df = pd.concat([df,df_temp])
df=df.drop(['entities','context_annotations','edit_history_tweet_ids','__twarc','attachments'],axis=1)
df.to_csv('/home/indy-stg2/mlteam1/processed_data.csv.bz2', compression='bz2',index=False)
#df.to_csv('/Users/alexzhu/Downloads/processed_data.csv.bz2', compression='bz2',index=False)