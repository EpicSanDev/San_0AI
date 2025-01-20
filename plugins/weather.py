import requests

def execute(location, api_key):
    """
    Plugin de météo pour San AI
    """
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': api_key,
        'units': 'metric',
        'lang': 'fr'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.ok:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity']
            }
    except:
        return None
