import pandas as pd
import numpy as np
import re 
import requests,json
from Preprocessing.Preprocessing import split_by_language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
lawyers_pere = pd.DataFrame(lawyers_response['lawyers'])
lawyers = lawyers_pere.rename(columns={
    "id":"lawyer_id"
}).drop(["email","union_number","affiliation_date","phone","agencies","issues","rates","avatar"],axis=1)
lawyers['years_of_experience'] = lawyers['years_of_experience'].apply(lambda x: f"{x}year")
# lawyers.head()

rates_response = rate_response.json()
rates = pd.DataFrame(rates_response["rates"])
rates.dropna(inplace=True)
rates.drop(["id"],axis=1,inplace=True)
rates['rating_rate'] = rates['rating_rate'].apply(lambda x :f"{x}star")
# rates.head()

agencies_response = agencie_response.json()
agencies = pd.DataFrame(agencies_response["data"]["agencies"]).rename(columns={
    "id":"agency_id"
})
agencies = agencies[["agency_id","lawyer_id"]]
# agencies.head()

issues_response = issue_response.json()
issues = pd.DataFrame(issues_response["issues"]).drop(["base_number","record_number"],axis=1)
# issues.head()

lawyers_with_rates = lawyers.merge(rates,on="lawyer_id")
# lawyers_with_rates.head()

lawyers_with_rates= pd.merge(agencies,lawyers_with_rates,on="lawyer_id",how="inner")
# lawyers_with_rates.head()

lawyers_with_rates = pd.merge(issues,lawyers_with_rates,on="agency_id",how="inner")
# lawyers_with_rates.head()

lawyers_with_rates["Text"] = lawyers_with_rates[['court_name','type','status','issue activity','name','address','union_branch','years_of_experience','rating_rate','estimated_cost']].astype(str).agg(" ".join,axis=1)
# lawyers_with_rates.head()
    
lawyers_with_rates['processed_text'] = lawyers_with_rates['Text'].apply(split_by_language)
# lawyers_with_rates.head()

documents = lawyers_with_rates[['lawyer_id','processed_text']].copy()
documents['processed_text'] = documents['processed_text'].apply(lambda x:" ".join(x))
documents.rename(columns={
    "lawyer_id":"docno",
    "processed_text":"text"
},inplace=True)
# documents.head()


index_path = r'C:\Users\Mohammad Kher\Desktop\Projects\IR\lawyer_index'
indexer = pt.DFIndexer(index_path,overwrite=True)

index_ref = indexer.index(documents['text'],documents['docno'].astype(str))

print("Index created with reference:", index_ref)





def Vectorize_cosine_similarity(user_input,document):
    all_descriptions = [user_input]+ documents["text"].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    user_vectors = tfidf_matrix[0]
    lawyer_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vectors, lawyer_vectors).flatten()
    ranked_lawyers = np.argsort(similarities)[::-1]
    return ranked_lawyers

def get_recommendation(ranked_lawyers):
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame()

    # Set to keep track of processed lawyer_ids
    processed_lawyer_ids = set()

    # Iterate through ranked_lawyers
    for idx in ranked_lawyers:
        lawyer_id = documents.iloc[idx]["docno"]
        
        # Check if the lawyer_id has already been processed
        if lawyer_id not in processed_lawyer_ids:
            # Filter the lawyers DataFrame for the current lawyer_id
            matched_lawyer = lawyers_pere[lawyers_pere['id'] == lawyer_id]
            
            # Append the matching row to the result DataFrame
            result = pd.concat([result, matched_lawyer], ignore_index=True)
            
            # Mark the lawyer_id as processed
            processed_lawyer_ids.add(lawyer_id)

    # return the resulting DataFrame
    return convert2json(result)

def convert2json(DataFrame):
    lawyers_data=[]
    final_rate = pd.DataFrame(DataFrame)
    for idx, row in DataFrame.iterrows():
        lawyer = {
            'id': row['id'],
            'name': row['name'],
            'email': row['email'],
            'address': row['address'],
            'union_branch': row['union_branch'],
            'union_number': row['union_number'],
            'affiliation_date': row['affiliation_date'],
            'specializations': row['specializations'],  # Assuming specialization is a single string in this example
            'years_of_experience': row['years_of_experience'],
            'description': row['description'],
            'phone': row['phone'],
            'avatar': row['avatar']
        }
    lawyers_data.append(lawyer)
    output_json = {
    'isSuccess': True,
    'lawyers': lawyers_data
    }
    json_output = json.dumps(output_json, indent=2, ensure_ascii=False)
    return json_output


