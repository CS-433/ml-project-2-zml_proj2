from tqdm.notebook import tqdm
import pandas as pd
import numpy as np


def add_to_data(tweet1, tweet2, data):    
    data['tweet_id1'].append(tweet1["id"])
    data['tweet_id2'].append(tweet2['id'])
    
    data['max_date'].append(max(tweet1['created_at'], tweet2['created_at']))
    data['min_date'].append(min(tweet1['created_at'], tweet2['created_at']))
    
    data['urls_count1'].append(tweet1['urls_count'])
    data['urls_count2'].append(tweet2['urls_count'])
    
    data['url_image_count1'].append(tweet1['url_image_count'])
    data['url_image_count2'].append(tweet2['url_image_count'])
    
    data['hashtags_count1'].append(tweet1['hashtags'])
    data['hashtags_count2'].append(tweet2['hashtags'])
    
    data['animated_gif_count1'].append(tweet1['animated_gif_count'])
    data['animated_gif_count2'].append(tweet2['animated_gif_count'])
    
    data['photo_count1'].append(tweet1['photo_count'])
    data['photo_count2'].append(tweet2['photo_count'])
    
    data['video_count1'].append(tweet1['video_count'])
    data['video_count2'].append(tweet2['video_count'])
    
    score1 = tweet1['like_count'] + tweet1['retweet_count'] + tweet1['reply_count']
    score2 = tweet2['like_count'] + tweet2['retweet_count'] + tweet2['reply_count']
    if  score1 > score2:
        data['winner'].append(0)
    else:
        data['winner'].append(1)


if __name__ == '__main__':
    """Run this script to group `similar` tweets into pairs, so they can be used
    for comparison in the BTM model. Tweets are considered similar under 2 conditions:
        1. Written by the same author
        2. Written within the 7 days (similar time)
    Mark the winning tweet: the one with higher engagement (sum of number of likes, retweets, and replys). 
    For noise reduction purposes, we don't create pairs between tweets that don't have at least differnece of 10
    in engagement, or at least 10% (whatever is higher)
    """
    # ===== Load data and Group authors ===== #
    tweets = pd.read_pickle('../data/tweets.pkl.bz2', compression='bz2') # sorted tweet data by time 
    authors = tweets.groupby("author_id") 
    # fillter out authors with less than 1 tweet as they can't form any pairs
    authors = authors.filter(lambda x : x.shape[0] > 1).groupby("author_id") 

    data = {"author": [], "author_followers_count": [], "verified": [],
        "tweet_id1": [], "urls_count1": [], "url_image_count1": [], 
        "hashtags_count1": [], "animated_gif_count1": [], "photo_count1": [], 
        "video_count1": [], "tweet_id2": [], "urls_count2": [], "url_image_count2": [],
        "hashtags_count2": [], "animated_gif_count2": [], "photo_count2": [], 
        "video_count2": [], "max_date": [], "min_date": [], "winner": []}

    # used for statistics purposes, saves the number of tweets by author for last number of authors
    shape = [] 

    # ===== Main loop to create pairs ===== #
    for i, author in tqdm(enumerate(authors)):
        shape.append(author[1].shape[0])
        for index1 in range(author[1].shape[0]):            
            tweet1 = author[1].iloc[index1] # get a tweet one from author
            for index2 in range(index1 + 1, min(author[1].shape[0], index1 + 1000)):
                tweet2 = author[1].iloc[index2] # get a tweet from author 
                    
                if (tweet1['created_at'] - tweet2['created_at']).days >= 7:
                    '''if there is 7 or more days of distance between them: break
                    as tweets are sorted by time, if tweet2 is to far from 
                    tweet1, then the next one can only be even further away
                    ''' 
                    break
                
                # see if there is enough difference in engagement between tweets to from a pair
                score1 = tweet1['like_count'] + tweet1['retweet_count'] + tweet1['reply_count']
                score2 = tweet2['like_count'] + tweet2['retweet_count'] + tweet2['reply_count']
                maxi = max(score1, score2)
                mini = max(1, min(score1, score2))
                if (maxi - mini) < 10 or maxi / mini < 1.1:
                    continue
                
                # save author's metadata
                data["author"].append(author[0])
                data["author_followers_count"].append(tweet1["author_followers_count"])
                data["verified"].append(tweet1["verified"])
                
                # randomize adding which tweet goes first as their order is by default sorted by creation time
                # this way we balance winner feature of the paired data
                if np.random.uniform() >= 0.5:
                    add_to_data(tweet1, tweet2, data)
                else:
                    add_to_data(tweet2, tweet1, data)
        
        # print statistics every 50000 iterations
        if i % 50_000 == 0:
            try:
                arr = np.array(shape)
                print(i, len(data["author"]), "Avg. shape", round(arr.mean(), 1), "Max", arr.max())
                shape = []
            except ValueError: 
                print(arr)

    # save the data as a dataframe
    data = pd.DataFrame(data)
    data.to_pickle("../data/pairs10%.pkl.bz2", compression='bz2')