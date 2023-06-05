import requests

API_BASE = 'https://api.nal.usda.gov/fdc/v1'
API_KEY = 'Ikxyb1yOtzhXjK8mtbYX8Gc9n7VerEG8xsNkw4ef'


def search_foods(query):
    params = {'query': query, 'pageSize': 20, 'dataType': ['Survey (FNDDS)']}
    return requests.get(API_BASE + "/search", params=params, headers={'X-Api-Key': API_KEY})


def get_energy(r):
    if(r.status_code == 200):
        data = r.json()
        if(data['totalHits'] >= 1):
            return next(filter(lambda nutrient: nutrient['nutrientName'] == 'Energy', data['foods'][0]['foodNutrients']))
        
    return None
