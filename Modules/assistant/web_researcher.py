import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import hashlib
from typing import Dict, List

class WebResearcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
        self.cache_timeout = 3600  # 1 heure
        self.max_retries = 3
        self.credibility_checker = CredibilityChecker()
        self.content_analyzer = ContentAnalyzer()
        
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        cache_key = self._generate_cache_key(query)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if (datetime.now() - cached_result['timestamp']).seconds < self.cache_timeout:
                return cached_result['data'][:max_results]
        
        results = []
        apis = {
            'wikipedia': self._search_wikipedia,
            'duckduckgo': self._search_duckduckgo,
            'google_scholar': self._search_google_scholar
        }
        
        for api_name, api_func in apis.items():
            for attempt in range(self.max_retries):
                try:
                    api_results = api_func(query)
                    results.extend(self._clean_and_deduplicate(api_results))
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"Échec définitif pour {api_name}: {str(e)}")
                    continue
        
        sorted_results = self._sort_by_relevance(results, query)
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': sorted_results
        }
        
        # Analyse de crédibilité et pertinence
        for result in results:
            credibility_score = self.credibility_checker.check(result['url'])
            content_quality = self.content_analyzer.analyze(result['content'])
            result['credibility'] = credibility_score
            result['quality'] = content_quality
            
        # Tri par crédibilité et pertinence
        results.sort(key=lambda x: (x['credibility'] * 0.6 + x['quality'] * 0.4), reverse=True)
        
        return results[:max_results]
        
    def _generate_cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()
        
    def _clean_and_deduplicate(self, results: List[Dict]) -> List[Dict]:
        seen_urls = set()
        cleaned = []
        for result in results:
            if result.get('url') not in seen_urls:
                seen_urls.add(result.get('url'))
                cleaned.append(result)
        return cleaned
        
    def _sort_by_relevance(self, results: List[Dict], query: str) -> List[Dict]:
        query_terms = set(query.lower().split())
        for result in results:
            relevance_score = sum(1 for term in query_terms 
                                if term in result.get('title', '').lower())
            result['relevance'] = relevance_score
        return sorted(results, key=lambda x: x['relevance'], reverse=True)
        
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

class CredibilityChecker:
    def check(self, url: str) -> float:
        # Vérifie la crédibilité d'une source
        domain_score = self._check_domain_authority(url)
        ssl_score = self._check_ssl_certificate(url)
        return (domain_score + ssl_score) / 2

class ContentAnalyzer:
    def analyze(self, content: str) -> float:
        # Analyse la qualité du contenu
        readability = self._assess_readability(content)
        informativeness = self._assess_informativeness(content)
        return (readability + informativeness) / 2
