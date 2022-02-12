import pandas as pd
import numpy as np

def load(path):
    df = pd.read_csv(path)
    return df



def prior(df):
    num_entries = len(df)

    ham_prior = (df.groupby(["label"], as_index=False).count().iloc[0]["text"])/num_entries
    spam_prior = (df.groupby(["label"], as_index=False).count().iloc[1]["text"])/num_entries
    
    
    return ham_prior, spam_prior



# number of emails that contain word w/#total num of emails

def likelihood(df):
    ham_like_dict = {}
    spam_like_dict = {}

    total_spam_emails = 0
    total_ham_emails = 0

    for idx, row in df.iterrows():
        cname = row["label"]
        text = row["text"].split(" ")

        # This is to ensure that I only add each word once per email. If the word "hello" appears twice in one email, I use this
        # list to ensure that I only increment my dict at hello only once
        already_seen_words = []

        if(cname=="ham"):
        
            total_ham_emails += 1
            for word in text:
                if (word in ham_like_dict) and (word not in already_seen_words):
                    ham_like_dict[word] += 1
                elif ((word not in already_seen_words)):
                    ham_like_dict[word] = 1
                
                already_seen_words.append(word)
                
                    
        else:
            total_spam_emails += 1
            for word in text:
                if (word in spam_like_dict) and (word not in already_seen_words):
                    spam_like_dict[word] += 1
                elif ((word not in already_seen_words)):
                    spam_like_dict[word] = 1
                
                already_seen_words.append(word)
    

        # divide every term in each dict by the total # of emails to get probabilities

    for key in ham_like_dict:
        ham_like_dict[key] = ham_like_dict[key]/total_ham_emails
    for key in spam_like_dict:
        spam_like_dict[key] = spam_like_dict[key]/total_spam_emails


    # These dictionaries contain the probability of each word occuring at least once in an email
    # If there are 10 spam emails with the word "friend" in it, then spam["friend"] = 10/total_num_of_spam_emails

    return ham_like_dict, spam_like_dict



def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
    '''
    prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
    '''
    #ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
    ham_spam_decision = None

    text = text.split(" ")


    '''YOUR CODE HERE'''
    #ham_posterior = posterior probability that the email is normal/ham
    ham_posterior = None
    
    ham_conditional_probabilities = [ham_like_dict[word] if word in ham_like_dict else 0.0004 for word in text]
    ham_posterior = ham_prior * (np.prod(ham_conditional_probabilities))

    #spam_posterior = posterior probability that the email is spam
    spam_posterior = None
    
    spam_conditional_probabilities = [spam_like_dict[word] if word in spam_like_dict else 0.0004 for word in text]
    spam_posterior = spam_prior * (np.prod(spam_conditional_probabilities))
    
    if(spam_posterior <= ham_posterior):
        ham_spam_decision = 0
    else:
        ham_spam_decision = 1

    '''END'''
    return ham_spam_decision

def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
    '''
    Calls "predict"
    '''
    hh = 0 #true negatives, truth = ham, predicted = ham
    hs = 0 #false positives, truth = ham, pred = spam
    sh = 0 #false negatives, truth = spam, pred = ham
    ss = 0 #true positives, truth = spam, pred = spam
    num_rows = df.shape[0]
    for i in range(num_rows):
        roi = df.iloc[i,:]
        roi_text = roi.text
        roi_label = roi.label_num
        guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
        if roi_label == 0 and guess == 0:
            hh += 1
        elif roi_label == 0 and guess == 1:
            hs += 1
        elif roi_label == 1 and guess == 0:
            sh += 1
        elif roi_label == 1 and guess == 1:
            ss += 1
    
    acc = (ss + hh)/(ss+hh+sh+hs)
    precision = (ss)/(ss + hs)
    recall = (ss)/(ss + sh)
    return acc, precision, recall


if __name__ == "__main__":
    traindf = load("./TRAIN_balanced_ham_spam.csv")
    ham_prior, spam_prior = prior(traindf)
    ham_like_dict, spam_like_dict = likelihood(traindf)

    trainacc, trainprecision, trainrecall = metrics(ham_prior, spam_prior, ham_like_dict, spam_like_dict, traindf)

    testdf = load("./TEST_balanced_ham_spam.csv")
    testacc, testprecision, testrecall = metrics(ham_prior, spam_prior, ham_like_dict, spam_like_dict, testdf)

    print("Train: ",trainacc,trainprecision, trainrecall)
    print("Test: ",testacc,testprecision, testrecall)
    
