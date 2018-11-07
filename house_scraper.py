import requests
from bs4 import BeautifulSoup

page = requests.get('https://www.zillow.com/homes/for_sale/palo-alto_rb/')
soup = BeautifulSoup(page.text, 'html.parser')
import ipdb
ipdb.set_trace()
