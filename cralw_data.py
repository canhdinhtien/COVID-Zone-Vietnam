import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://vi.wikipedia.org/wiki/B%E1%BA%A3n_m%E1%BA%ABu:D%E1%BB%AF_li%E1%BB%87u_%C4%91%E1%BA%A1i_d%E1%BB%8Bch_COVID-19/Th%E1%BB%91ng_k%C3%AA_t%E1%BA%A1i_Vi%E1%BB%87t_Nam'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup)

# Try to find any table
tables = soup.find_all('table', {'class': 'wikitable'})

# Iterate through tables to find the correct one
for table in tables:
    cities = []
    infected_cases = []
    deaths = []
    new_cases = []

    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        
        if len(cells) >= 4:  # Ensure there are enough cells in the row
            city = cells[0].text.strip()
            infected_case = cells[1].text.strip()
            death = cells[2].text.strip()
            new_case = cells[3].text.strip()

            cities.append(city)
            infected_cases.append(infected_case)
            deaths.append(death)
            new_cases.append(new_case)

    df = pd.DataFrame({
        'City': cities,
        'Infected Cases': infected_cases,
        'Deaths': deaths,
        'New Cases': new_cases
    })

    df.iloc[1:].to_csv('Thong_tin_covid.csv', index=False)
    print('Luu thanh cong')
    break
else:
    print('Khong tim thay')