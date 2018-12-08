import uuid
import os
import sys
import csv
import requests
import time

import asyncio
import aiohttp        
import aiofiles

root_dir = '.'
if not os.path.exists('csvs'):
    os.makedirs('csvs')

def city_state_string(city, state):
    return city.replace(" ", '-') + '_' + state

def save_image_todo(city, state):
    cs_str = city_state_string(city, state)
    basedir = os.path.join(root_dir, cs_str)
    results = []
    dicts = []
    for pg in os.listdir(basedir):
        pgdir = os.path.join(basedir, pg)
        pgcsv = os.path.join(pgdir, 'data.csv')
        with open(pgcsv, 'r', encoding='cp1252') as f:
            reader = csv.DictReader(f)
            for row in reader:
                id = row['id']
                url = row['image_url']
                if (len(url) > 3 and url[-3:] == 'jpg'):
                    results.append((url, id + '.jpg'))
                    dicts.append(row)

    with open(cs_str + '.download', 'w') as f:
        for url, fname in results:
            f.write(f"{fname} {url}\n")

    csvout = os.path.join('csvs', cs_str + '.csv')
    fieldnames = ['id', 'price', 'price_currency', 'manufacturer', 'beds', 'baths', 'lot_size', 'image_url']
    with open(csvout, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in dicts:
            writer.writerow(d)

async def save_image(image_url, filename, session):
    async with session.get(image_url) as resp:
        image_url = image_url.replace('https', 'http')
        if resp.status == 200:
            f = await aiofiles.open(filename, mode='wb')
            await f.write(await resp.read())
            await f.close()
            print(f"Saved to {filename} from {image_url}")

async def download_images(loop, city, state, dirname='images'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cs_str = city_state_string(city, state)
    dfile = cs_str + '.download'
    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = []
        with open(dfile, 'r') as f:
            for line in f:
                fname, url = line.split()
                fname = os.path.join(dirname, fname)
                tasks.append(save_image(url, fname, session))
            await asyncio.gather(*tasks)

city_name, state_abbrev = 'Seattle', 'WA'
save_image_todo(city_name, state_abbrev)
loop = asyncio.get_event_loop()
loop.run_until_complete(download_images(loop, city_name, state_abbrev))
