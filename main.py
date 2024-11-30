import pandas as pd
import numpy as np
import pyterrier as pt
import requests,json
from Preprocessing.Preprocessing import split_by_language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if not pt.started():
    pt.init()


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


index_path = r'C:\Users\Mohammad Kher\Desktop\IR\lawyer_index'
indexer = pt.DFIndexer(index_path,overwrite=True)

index_ref = indexer.index(documents['text'],documents['docno'].astype(str))

print("Index created with reference:", index_ref)



def Vectorize_cosine_similarity(user_input,document=documents):
    all_descriptions = [user_input]+ documents["text"].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    user_vectors = tfidf_matrix[0]
    lawyer_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vectors, lawyer_vectors).flatten()
    ranked_lawyers = np.argsort(similarities)[::-1]
    return get_recommendation(ranked_lawyers)

def get_recommendation(ranked_lawyers):
    result = []
    processed_lawyer_ids = set()

    for idx in ranked_lawyers:
        lawyer_id = documents.iloc[idx]["docno"]
        
        # Ensure unique recommendations
        if lawyer_id not in processed_lawyer_ids:
            matched_lawyer = lawyers_pere[lawyers_pere['id'] == int(lawyer_id)]
            
            # Append matched lawyer data
            if not matched_lawyer.empty:
                result.append(matched_lawyer.iloc[0].to_dict())
                processed_lawyer_ids.add(lawyer_id)

    return convert2json(result)


def convert2json(lawyers_list):
    lawyers_data = []
    for row in lawyers_list:
        lawyer = {
            'id': row['id'],
            'name': row['name'],
            'email': row.get('email', None),
            'address': row.get('address', None),
            'union_branch': row.get('union_branch', None),
            'union_number': row.get('union_number', None),
            'affiliation_date': row.get('affiliation_date', None),
            'specializations': row.get('specializations', None),
            'years_of_experience': row.get('years_of_experience', None),
            'description': row.get('description', None),
            'phone': row.get('phone', None),
            'avatar': row.get('avatar', None),
        }
        lawyers_data.append(lawyer)

    output_json = {
        'isSuccess': True,
        'lawyers': lawyers_data
    }
    return output_json




if __name__ == "__main__":
    print(Vectorize_cosine_similarity("شرعية عائلية"))