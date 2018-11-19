import uuid
import os
import sys
import csv
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import time

import logging
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.ERROR)

import asyncio
import aiohttp        
import aiofiles
from arsenic import get_session, keys, browsers, services

realtor_base = 'https://www.realtor.com/'


def city_state_string(city, state):
    return city.replace(" ", '-') + '_' + state

def get_search_url(city, state, page=1):
    cs_str = city_state_string(city, state)
    page_str = ''
    if page > 1:
        page_str = '/pg-' + str(page)
    return realtor_base + 'realestateandhomes-search/' + cs_str + '/type-single-family-home' + page_str

def get_num_pages(source):
    sei = source.rfind('srp_list:paging:', 0, source.rfind('srp_list:paging:'))
    index = sei + len('srp_list:paging:')
    end = source.find('"', index)
    return int(source[index:end])


def ignore_exception(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None
    return new_func


class Scraper(object):
    def __init__(self, source, city, state, page, root_dir):
        self.source = source
        self.cs_str = city_state_string(city, state)
        self.city = city
        self.state = state
        self.page = page
        self.page_str = "%04d" % page
        self.root_dir = root_dir
        self.imgsave_semaphore = asyncio.Semaphore(5)
        self.setup_directories()
        print(f'Initialized Scraper for {city}, {state} to load page {page}')
        print(f'Will save to directory: {self.dir}')
        
    def setup_directories(self):
        self.dir = os.path.join(self.root_dir, self.cs_str, self.page_str)
        self.imgdir = os.path.join(self.dir, 'images')
        if not os.path.exists(self.imgdir):
            os.makedirs(self.imgdir)
    
    @ignore_exception
    def get_price(self, house):
        return int(house.find('meta', {'itemprop': 'price'})['content'])
    
    @ignore_exception
    def get_price_currency(self, house):
        return house.find('meta', {'itemprop': 'priceCurrency'})['content']
    
    @ignore_exception
    def get_manufacturer(self, house):
        return house.find('meta', {'itemprop': 'manufacturer'})['content']
    
    @ignore_exception
    def get_beds(self, house):
        return int(house.find('li', {'data-label': "property-meta-beds"})
                   .find('span').contents[0].replace(',', ''))
    
    @ignore_exception
    def get_baths(self, house):
        return int(house.find('li', {'data-label': "property-meta-baths"})
                   .find('span').contents[0].replace(',', ''))
    
    @ignore_exception
    def get_lot_size(self, house):
        return int(house.find('li', {'data-label': "property-meta-lotsize"})
                   .find('span').contents[0].replace(',', ''))
    
    @ignore_exception
    def get_image_url(self, house):
        return house.find('img')['src'] or house.find('img')['data-src']
    
    def extract_data(self, house):
        data = {
            'id': uuid.uuid4().hex,
            'price': self.get_price(house),
            'price_currency': self.get_price_currency(house),
            'manufacturer': self.get_manufacturer(house),
            'beds': self.get_beds(house),
            'baths': self.get_baths(house),
            'lot_size': self.get_lot_size(house),
            'image_url': self.get_image_url(house)
        }
        return data
    
    async def save_image(self, image_url, filename, session):
        async with self.imgsave_semaphore:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(filename, mode='wb')
                    await f.write(await resp.read())
                    await f.close()
                        
    def scrape_all(self):
        self.soup = BeautifulSoup(self.source, 'html.parser')
        self.houses = self.soup.find_all('li', class_='component_property-card')
        csv_fname = os.path.join(self.dir, 'data.csv')
        fieldnames = ['id', 'price', 'price_currency', 'manufacturer', 'beds', 'baths', 'lot_size', 'image_url']
        with open(csv_fname, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for house in self.houses:
                data = self.extract_data(house)
                writer.writerow(data)


def load(city, state):
    # Selenium: base implementation for loading
    driver = webdriver.PhantomJS(service_log_path='./ghostdriver2.log')
    print(city, state)
    print(get_search_url(city, state))
    driver.get(get_search_url(city, state))
    num_pages = get_num_pages(driver.page_source)

    print(f'Loading {num_pages} pages for {city}, {state}')
    for i in range(num_pages):
        driver.get(get_search_url(city, state, i+1))
        scraper = Scraper(driver.page_source, city, state, i+1, '.')
        scraper.scrape_all()
        time.sleep(5)


if __name__ == '__main__':
    load('Houston', 'TX')
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(load(loop, 'New York City', 'NY'))