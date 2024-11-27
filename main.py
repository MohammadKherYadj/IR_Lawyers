import pandas as pd
import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import requests

from Preprocessing.Preprocessing_EN import Preprocessing_EN
from Preprocessing.Preprocessing_AR import Preprocessing_AR


# get lawyers from database
lawyer_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/lawyers")
print(f"status code : {lawyer_response.status_code}")


# get rates from database
rate_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/rates")
print(f"status code : {rate_response.status_code}")


# get agencies from database
agencie_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/agencies")
print(f"status code : {agencie_response.status_code}")


# get issues from database
issue_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/issues")
print(f"status code : {issue_response.status_code}")


lawyers_response = lawyer_response.json()
lawyers = pd.DataFrame(lawyers_response['lawyers']).rename(columns={
    "id":"lawyer_id"
}).drop(["email","union_number","affiliation_date","phone","agencies","issues","rates"],axis=1)
lawyers['years_of_experience'] = lawyers['years_of_experience'].apply(lambda x: f"{x}year")
# lawyers.head()

rates_response = rate_response.json()
rates = pd.DataFrame(rates_response["rates"])
rates.dropna(inplace=True)
rates.drop(["id"],axis=1,inplace=True)
rates['rating_rate'] = rates['rating_rate'].apply(lambda x :f"{x}star")
# rates.head()

agencies_response = agencie_response.json()
agencies = pd.DataFrame(agencies_response["agencies"]).rename(columns={
    "id":"agency_id"
})
agencies = agencies[["agency_id","lawyer_id"]]
# agencies.head()

issues_response = issue_response.json()
issues = pd.DataFrame(issues_response["issues"]).drop(["base_number","record_number"],axis=1)
# issues.head()

lawyers_with_rates = lawyers.merge(rates,on="lawyer_id")
# lawyers_with_rates.head()

lawyers_with_rates= pd.merge(agencies,lawyers_with_rates,on=["lawyer_id"],how="inner")
# lawyers_with_rates.head()

lawyers_with_rates = pd.merge(issues,lawyers_with_rates,on="agency_id",how="inner")
# lawyers_with_rates.head()

lawyers_with_rates["Text"] = lawyers_with_rates[['court_name','type','status','issue activity','name','address','union_branch','years_of_experience','rating_rate','estimated_cost']].astype(str).agg(" ".join,axis=1)
# lawyers_with_rates.head()

def preprocessing(text,language):
    if language == "en":
        return Preprocessing_EN.process(text)
    else:
        return Preprocessing_AR.process(text)
    
lawyers_with_rates['processed_text'] = lawyers_with_rates.apply(lambda x: preprocessing(x['description'], x['language']), axis=1)
# lawyers_with_rates["Text"]


    