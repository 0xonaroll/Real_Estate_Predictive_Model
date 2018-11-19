import uuid
import os
import sys
import csv
import requests
from selenium import webdriver
from bs4 import BeautifulSoup

import logging
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.ERROR)

import asyncio
import aiohttp        
import aiofiles
from arsenic import get_session, keys, browsers, services

realtor_base = 'https://54.148.36.182/'


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
    def __init__(self, loop, service, browser, city, state, page, root_dir, master_semaphore):
        self.loop = loop
        self.service = service
        self.browser = browser
        self.cs_str = city_state_string(city, state)
        self.city = city
        self.state = state
        self.page = page
        self.page_str = "%04d" % page
        self.root_dir = root_dir
        self.imgsave_semaphore = asyncio.Semaphore(5)
        self.master_semaphore = master_semaphore
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
                        
    async def scrape_all(self):
        async with self.master_semaphore:
            async with get_session(self.service, self.browser) as sess:
                await sess.get(get_search_url(self.city, self.state, self.page))
                self.source = await sess.get_page_source()
                print('received source')
                self.soup = BeautifulSoup(self.source, 'html.parser')
                self.houses = self.soup.find_all('li', class_='component_property-card')

                csv_fname = os.path.join(self.dir, 'data.csv')
                fieldnames = ['id', 'price', 'price_currency', 'manufacturer', 'beds', 'baths', 'lot_size', 'image_url']
                print('scraping all')
                with open(csv_fname, mode='w') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    async with aiohttp.ClientSession(loop=self.loop) as session:
                        tasks = []
                        print(len(self.houses))
                        for house in self.houses:
                            data = self.extract_data(house)
                            fname = os.path.join(self.imgdir, data['id'] + '.jpg')
                            tasks.append(self.save_image(data['image_url'], fname, session))
                            writer.writerow(data)
                        await asyncio.gather(*tasks)
                        print('gathered tasks')


async def load(loop, city, state):
    master_semaphore = asyncio.Semaphore(5)
    service = services.Chromedriver()
    browser = browsers.Chrome(chromeOptions={
        'args': ['--headless', '--disable-gpu', '--log-level=3', '--disable-logging']
    })

    # Selenium: base implementation for loading
    # driver = webdriver.PhantomJS(service_log_path='./ghostdriver2.log')
    # driver.get(get_search_url(city, state))
    # async with get_session(self.service, self.browser) as sess:
    # print(driver.page_source)
    # num_pages = get_num_pages(driver.page_source)

    num_pages = 2

    print(f'Loading {num_pages} pages for {city}, {state}')
    tasks = []
    for i in range(num_pages):
        scraper = Scraper(loop, service, browser, city, state, i+1, '.', master_semaphore)
        tasks.append(scraper.scrape_all())
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load(loop, 'New York City', 'NY'))