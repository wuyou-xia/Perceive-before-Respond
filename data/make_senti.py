from cnsenti import Sentiment
import json
import pickle
from tqdm import tqdm

anno_root = 'release_train.json'
annos = []
senti = Sentiment()
with open(anno_root, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        anno = json.loads(line)
        anno['sentiment'] = []
        for context in anno['context']:
            text = context['text'].replace(' ', '')
            text_sentiment = senti.sentiment_calculate(text)
            anno['sentiment'].append(text_sentiment)
        annos.append(anno)

## save
with open('release_train_senti.pkl', 'wb') as f:
    pickle.dump(annos, f)
