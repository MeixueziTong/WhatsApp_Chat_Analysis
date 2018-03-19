
# coding: utf-8

# import modules


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')



# read files


import re
pattern = r'\[(\d+\/\d+\/\d+,\s\d+:\d+:\d+)\]\s(\w+)\:(.*)'
with open('_chat.txt','r') as f:
    chats = re.findall(pattern, f.read())
    
chats = pd.DataFrame(chats, columns = ['DateTime', 'Sender', 'Msg'])
chats.to_csv('chats.csv', index = False)


# parse datetime column


from datetime import datetime
from dateutil.parser import parse

chats['DateTime'] = [parse(x,dayfirst= True) for x in chats['DateTime']]


# check total number of chats

print('The total number of chats is', len(chats))


# check the chat contents:


print(chats.head(2))


# chat statistics:


start = chats['DateTime'][0]
end = chats['DateTime'][len(chats)-1]
chat_period = end-start
total_chats = len(chats)
average_chat_num = total_chats/chat_period.days
print('We have been talking on Whatsapp since {}. It has been {} days till {}.\n'.format(start, chat_period.days, end)) 
print('We have sent each other {} messages in total, on average {:.0f} messages a day.'.format(total_chats, average_chat_num))


# chat cleaning and preprocessing:


from nltk.tokenize import word_tokenize
#eliminate all non-alphabetical characters, hyperlinks, and stopwords, omitted image
from nltk.corpus import stopwords
stopWords = stopwords.words('english') # remove stopwords
dropWords = ['','omitted','image','yeah','sounds','good','ok','sure','like'] + stopWords # remove common response words
def preprocess(chats):
    message = []
    for chat in chats:
        chat = word_tokenize(chat)
        chat = [token.lower() for token in chat]
        chat = [re.sub(r'[\d\W\?\.,!]','',token) for token in chat]
        chat = [re.sub(r'\bnt\b', 'not', token) for token in chat]
        chat = [token for token in chat if token not in dropWords]
        message.append(chat)
    return message
chat_copy = chats['Msg'].copy
chat_copy = preprocess(chats['Msg'])
chats['Msg'] = chat_copy    


# check preprocessing results:

chats['Msg'].head(3)




Michelle = chats[chats['Sender'] == 'Michelle']
Matt = chats[chats['Sender'] == 'Matt']
print('Michelle has sent {} messages to Matt; Matt has sent {} messages to Michelle.'.format(len(Michelle), len(Matt)))




from collections import Counter
def mergeMsg(Msg):
    mergedMsg = []
    for msg in Msg:
        mergedMsg += msg
    return mergedMsg




from collections import Counter
Matt_Msg = mergeMsg(Matt['Msg'])
Matt_word_counts = Counter(Matt_Msg)
Michelle_Msg = mergeMsg(Michelle['Msg'])
Michelle_word_counts = Counter(Michelle_Msg)


# Compute wordcloud


from wordcloud import WordCloud

# lower max_font_size
wordcloud_Michelle = WordCloud(max_font_size=50).generate_from_frequencies(Michelle_word_counts)
wordcloud_Matt = WordCloud(max_font_size=50).generate_from_frequencies(Matt_word_counts)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
ax1.imshow(wordcloud_Michelle, interpolation="bilinear")
ax2.imshow(wordcloud_Matt, interpolation="bilinear")
ax1.axis("off")
ax2.axis("off")
ax1.set_title("Michelle to Matt")
ax2.set_title("Matt to Michelle")
plt.savefig('word_cloud.png', dpi = 300,  bbox_inches='tight')




ts = pd.Series(np.ones(len(chats)), index = chats['DateTime'])
ts_michelle = pd.Series(np.ones(len(Michelle)), index = Michelle['DateTime'])
ts_matt = pd.Series(np.ones(len(Matt)), index = Matt['DateTime'])

                                        


# plot message counts by month:


month_michelle = ts_michelle.resample('M', kind = 'period').sum() # aggregate data by month
month_matt = ts_matt.resample('M', kind = 'period').sum()
plt.figure()
ax1 = month_michelle.plot(label = 'Michelle')
ax2 = month_matt.plot(label = 'Matt')
plt.ylabel('message counts')
plt.title('message counts by month')
plt.xlabel('')
plt.legend()
plt.savefig('message_counts_by_month.png')


# plot message counts by hour of the day:


time =[ datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").hour for x in chats['DateTime']] 
#time = chats['DateTime'].values.time()
hour_counts = np.array([x for x in Counter(time).items()])
for i in range(23):
    if i not in list(hour_counts[:,0]):
        hour_counts = np.append(hour_counts, [[i, 0]], axis = 0)
hour_counts = np.sort(hour_counts, axis =0)
hour_counts_pd = pd.Series(hour_counts[:,1], index = hour_counts[:,0])

plt.figure()
hour_counts_pd.plot.bar()
plt.xlabel('hours of a day')
plt.xticks(rotation = 45)
plt.ylabel('message counts')
plt.title('message counts  by hour')
plt.savefig('message_counts_by_hour.png')

