import requests
from bs4 import BeautifulSoup

trulia_url = 'https://www.trulia.com/NY/New_York/'
realtor_url = 'https://www.realtor.com/realestateandhomes-search/Sugar-Land_TX'
# page = requests.get('https://www.zillow.com/homes/for_sale/palo-alto_rb/')
page = requests.get(realtor_url)
soup = BeautifulSoup(page.text, 'html.parser')
import ipdb
ipdb.set_trace()
