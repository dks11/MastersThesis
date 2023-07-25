import sys
import pandas as pd
import random
import numpy as np
import random
import os
import csv

#EnglishFrench/Russian
Englishdf1 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\enron1_prepared_train.jsonl", lines = True, encoding='utf-8')
Englishdf2 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\enron1_prepared_valid.jsonl", lines = True, encoding='utf-8')
Englishdf = pd.concat([Englishdf1,Englishdf2])


dfHamEnglish = Englishdf[Englishdf['completion'] == 0]
dfHamEnglish = dfHamEnglish.sample(n = 2517)
dfSpamEnglish = Englishdf[Englishdf['completion'] == 1]
dfSpamEnglish = dfSpamEnglish.sample(n = 1028)

Frenchdf1 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\FrenchPrompts_prepared_train.jsonl", lines = True, encoding='utf-8')
Frenchdf2 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\FrenchPrompts_prepared_valid.jsonl", lines = True, encoding='utf-8')
Frenchdf = pd.concat([Frenchdf1,Frenchdf2])

dfHamFrench = Frenchdf[Frenchdf['completion'] == 0]
dfHamFrench = dfHamFrench.sample(n = 219)
dfSpamFrench = Frenchdf[Frenchdf['completion'] == 1]
dfSpamFrench = dfSpamFrench.sample(n = 219)

f = open("EnglishFrenchTrainDownsample.jsonl", "w", encoding='utf-8')
finaldf = pd.concat([dfHamEnglish,dfHamFrench,dfSpamEnglish, dfSpamFrench])
finaldf.to_json(f, lines = True, orient="records", force_ascii = False)



#EnglishRussian/French
Englishdf1 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\enron1_prepared_train.jsonl", lines = True, encoding='utf-8')
Englishdf2 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\enron1_prepared_valid.jsonl", lines = True, encoding='utf-8')
Englishdf = pd.concat([Englishdf1,Englishdf2])


dfHamEnglish = Englishdf[Englishdf['completion'] == 0]
dfHamEnglish = dfHamEnglish.sample(n = 2715)
dfSpamEnglish = Englishdf[Englishdf['completion'] == 1]
dfSpamEnglish = dfSpamEnglish.sample(n = 1109)

Russiandf1 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\RussianPrompts_prepared_train.jsonl", lines = True, encoding='utf-8')
Russiandf2 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\RussianPrompts_prepared_valid.jsonl", lines = True, encoding='utf-8')
Russiandf = pd.concat([Russiandf1,Russiandf2])

dfHamRussian = Russiandf[Russiandf['completion'] == 0]
dfHamRussian = dfHamRussian.sample(n = 80)
dfSpamRussian = Russiandf[Russiandf['completion'] == 1]
dfSpamRussian = dfSpamRussian.sample(n = 79)

f = open("EnglishRussianTrainDownsample.jsonl", "w", encoding='utf-8')
finaldf = pd.concat([dfHamEnglish,dfHamRussian,dfSpamEnglish, dfSpamRussian])
finaldf.to_json(f, lines = True, orient="records", force_ascii = False)



#FrenchRussian/English
Frenchdf1 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\FrenchPrompts_prepared_train.jsonl", lines = True, encoding='utf-8')
Frenchdf2 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\FrenchPrompts_prepared_valid.jsonl", lines = True, encoding='utf-8')
Frenchdf = pd.concat([Frenchdf1,Frenchdf2])


dfHamFrench = Frenchdf[Frenchdf['completion'] == 0]
dfHamFrench = dfHamFrench.sample(n = 173)
dfSpamFrench = Frenchdf[Frenchdf['completion'] == 1]
dfSpamFrench = dfSpamFrench.sample(n = 172)

Russiandf1 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\RussianPrompts_prepared_train.jsonl", lines = True, encoding='utf-8')
Russiandf2 = pd.read_json("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Prompt Files\\RussianPrompts_prepared_valid.jsonl", lines = True, encoding='utf-8')
Russiandf = pd.concat([Russiandf1,Russiandf2])

dfHamRussian = Russiandf[Russiandf['completion'] == 0]
dfHamRussian = dfHamRussian.sample(n = 64)
dfSpamRussian = Russiandf[Russiandf['completion'] == 1]
dfSpamRussian = dfSpamRussian.sample(n = 63)

f = open("FrenchRussianTrainDownsample.jsonl", "w", encoding='utf-8')
finaldf = pd.concat([dfHamFrench,dfHamRussian,dfSpamFrench, dfSpamRussian])
finaldf.to_json(f, lines = True, orient="records", force_ascii = False)

values = """
-------------------------------------------
EnglishFrench/Russian

EnFr = 5570
EnFr Ham % = 69
EnFr Spam % = 31
English % = 89
French % = 11

English = 4979
English Ham % = 71
English Spam % = 29

French = 591
French Ham % = 50
French Spam % 50

Downsampled EnglishFrench
Size = 3983 (Size of English Train)
English = 3545
English Ham = 2517
English Spam = 1028
French = 438
French Ham = 219
French Spam = 219

----------------------------------------------

EnglishRussian/French

EnRu = 5198
EnRu Ham % = 70
EnRu Spam % = 30
English % = 96
Russian % = 4


English = 4979
English Ham % = 71
English Spam % = 29

Russian = 219
Russian Ham % = 50
Russian Spam % 50


Downsampled EnglishRussian
Size = 3983 (Size of English Train)
English = 3824
English Ham = 2715
English Spam = 1109
Russian = 159
Russian Ham = 80
Russian Spam = 79

------------------------------------------------

FrenchRussian/English

FrRu = 810
FrRu Ham % = 50 
FrRu Spam % = 50
French % = 73
Russian % = 27


French = 591
French Ham % = 50
French Spam % 50

Russian = 219
Russian Ham % = 50
Russian Spam % 50

Downsampled FrenchRussian
Size = 472 (Size of French Train)
French = 345
French Ham = 173
French Spam = 172
Russian = 127
Russian Ham = 64
Russian Spam = 63
"""