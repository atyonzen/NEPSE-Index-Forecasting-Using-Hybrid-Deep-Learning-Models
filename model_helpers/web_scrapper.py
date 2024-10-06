# pip install playwright
# playwright install

import asyncio
import numpy as np
import re
from playwright.async_api import async_playwright
import random
from variable_config import file_name

# List of dictionaries of windows based custom userAgents
win_user_agents = [
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.3"},
    {"ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.3"},
    {"ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.1"},
    {"ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.3"},
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120."},
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 OPR/105.0.0."},
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121."},
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0."},
    {"ua": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115."},
    {"ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.3"},
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.3"}
]

# Function for extracting table data
async def extract_table_data(scraped_data, page):
    # Extract the table data
    rows = await page.query_selector_all('.table-responsive tbody tr')

    # Loop through each row and extract the data
    for row in rows:
        columns = await row.query_selector_all('td')
        row_data = [re.sub(r"[ ,]", "", await cell.inner_text()) for cell in columns]
        scraped_data.append(row_data)
    
    return scraped_data

async def save_to_csv(file_name, data):
    # Save the scraped data to a CSV file
    np.savetxt(f'{file_name}.csv', data, delimiter=',', fmt='%s')


async def scrape_nepse_indices():
    async with async_playwright() as p:
        # Launch a new browser
        # Headless mode required custom userAgent
        browser = await p.chromium.launch(headless=True)  # Set headless=False if you want to see the browser
        context = await browser.new_context(user_agent=random.choice(win_user_agents)['ua'])
        page = await context.new_page()
        # page = await browser.new_page()

        try:
            # Go to the NEPSE index page
            await page.goto('https://www.nepalstock.com.np/indices', wait_until='networkidle')
        except Exception as e:
            print(f'Error loading page: {e}')
            return  # Handle the error, log it, or retry

        # Wait for the necessary content to load
        await page.wait_for_selector('.table-responsive thead')  # Wait for the table to be present

        next_button_visible = True
        page_counter = 1
        scraped_data = []
        if ( page_counter == 1):
            # Extract the table headers
            headers = await page.query_selector_all('.table-responsive thead tr th')
            headers_text = [(await header.inner_text()).replace("'", "") for header in headers]
            scraped_data.append(headers_text)

        while (next_button_visible):
            # increment the page counter
            page_counter+=1

            # Wait for the table to appear
            await page.wait_for_selector('.table-responsive')

            # Locate the next button in the pagination
            navigation = page.locator('li.pagination-next')

            # Check if the next button is disabled
            is_disabled = await navigation.get_attribute('class')

            if 'disabled' not in is_disabled:
                await navigation.click()

                # Wait for the page to load completely before moving to the next page
                await page.wait_for_load_state('networkidle')

                # Extract the table data
                await extract_table_data(scraped_data, page)

                next_button_visible = True
            else:
                # No more pages; exit the loop
                next_button_visible = False

        # Close the browser
        await browser.close()

        # # Save the scraped data to a CSV file
        await save_to_csv(f'./data_src/{file_name}', scraped_data)
        # np.savetxt('nepse_indices.csv', scraped_data, delimiter=',', fmt='%s')
        # print("Data saved to nepse_indices.csv")

# Run the async function
asyncio.run(scrape_nepse_indices())