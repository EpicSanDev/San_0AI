import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

class WebResearcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
        
    def search(self, query, max_results=5):
        # Recherche via différentes API
        results = []
        apis = {
            'wikipedia': self._search_wikipedia,
            'duckduckgo': self._search_duckduckgo
        }
        
        for api_name, api_func in apis.items():
            try:
                results.extend(api_func(query))
            except Exception as e:
                print(f"Erreur avec {api_name}: {str(e)}")
                
        return results[:max_results]
        
    def _search_wikipedia(self, query):
        url = f"https://fr.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query
        }
        response = self.session.get(url, params=params, headers=self.headers)
        return response.json()['query']['search']
