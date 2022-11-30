#!/usr/bin/env python
# coding: utf-8

# # <center>HW2: Web Scraping</center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work or let someone copy your solution (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. No last minute extension of due date. Be sure to start working on it ASAP! </div>

# ## Q1. Collecting Movie Reviews
# 
# Write a function `getReviews(url)` to scrape all **reviews on the first page**, including, 
# - **title** (see (1) in Figure)
# - **reviewer's name** (see (2) in Figure)
# - **date** (see (3) in Figure)
# - **rating** (see (4) in Figure)
# - **review content** (see (5) in Figure. For each review text, need to get the **complete text**.)
# - **helpful** (see (6) in Figure). 
# 
# 
# Requirements:
# - `Function Input`: book page URL
# - `Function Output`: save all reviews as a DataFrame of columns (`title, reviewer, rating, date, review, helpful`). For the given URL, you can get 24 reviews.
# - If a field, e.g. rating, is missing, use `None` to indicate it. 
# 
#     
# ![alt text](IMDB.png "IMDB")

# In[1]:


import requests

# Add your import statements
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions 
import time


# In[2]:



def getReviews(page_url):
    
    reviews = None
   
    # Add your code here
    
    page = requests.get(page_url)
    
    if page.status_code==200:  
        soup = BeautifulSoup(page.content, 'html.parser')
        
    reviews = pd.DataFrame(columns = ['title','reviewer','date','rating','review','helpful'])
    
    divs = soup.select('div.lister-list div.imdb-user-review')
    
    for idx,div in enumerate(divs):
        
        title = None
        reviewer = None
        date = None
        review = None
        helpful = None
        rating =  None
        
        s_title = div.find('a')
        if s_title!=None:
            title= s_title.get_text()
            
        s_reviewer = div.find('span', class_ = 'display-name-link')
        if s_reviewer!=None:
            reviewer=s_reviewer.get_text()
            
            
        s_date = div.find('span', class_ = 'review-date')
        if s_date!=None:
            date=s_date.get_text()
            
            
        s_rating = div.find('span', class_ = 'rating-other-user-rating')
        if s_rating!=None:
            rating=s_rating.span.get_text()
            
            
        s_review = div.find('div', class_ = 'show-more__control')
        if s_review!=None:
            review=s_review.get_text()
            
            
        s_helpful = div.find('div', class_='text-muted')
        if s_helpful!=None:
            helpful=s_helpful.get_text().split('\n')[1].strip()


        reviews.loc[idx]=(title,reviewer,date,rating,review,helpful)
        
    
    return reviews


# In[3]:


# Test your function

page_url = 'https://www.imdb.com/title/tt1745960/reviews?sort=totalVotes&dir=desc&ratingFilter=0'
reviews = getReviews(page_url)

print(len(reviews))
print(reviews.head())


# ## Q2 (Bonus) Collect Dynamic Content
# 
# Write a function `get_N_review(url, webdriver)` to scrape **at least 100 reviews** by clicking "Load More" button 5 times through Selenium WebDrive, 
# 
# 
# Requirements:
# - `Function Input`: book page `url` and a Selenium `webdriver`
# - `Function Output`: save all reviews as a DataFrame of columns (`title, reviewer, rating, date, review, helpful`). For the given URL, you can get 24 reviews.
# - If a field, e.g. rating, is missing, use `None` to indicate it. 
# 
# 

# In[4]:


def getReviews(page_url, driver):
    
    reviews = None
    driver.get(page_url)
    
    # add your code here
    
    reviews = None
    driver.get(page_url)
    load_more = driver.find_element(By.CSS_SELECTOR,"button#load-more-trigger")

    for i in range(5):
        load_more.click()
        time.sleep(3)
    
    page = driver.page_source
    
    soup = BeautifulSoup(page, 'html.parser')
    
    reviews = pd.DataFrame(columns = ['title','reviewer','date','rating','review','helpful'])
    
    divs = soup.select('div.lister-item.imdb-user-review')
    
    for idx,div in enumerate(divs):
        
        title = None
        reviewer = None
        date = None
        review = None
        helpful = None
        rating =  None
        
        
        s_title = div.find('a')
        if s_title!=None:
            title= s_title.get_text()
            
        s_reviewer = div.find('span', class_ = 'display-name-link')
        if s_reviewer!=None:
            reviewer=s_reviewer.get_text()
            
            
        s_date = div.find('span', class_ = 'review-date')
        if s_date!=None:
            date=s_date.get_text()
            
            
        s_rating = div.find('span', class_ = 'rating-other-user-rating')
        if s_rating!= None:
            rating=s_rating.span.get_text()
            
            
        s_review = div.find('div', class_ = 'show-more__control')
        if s_review!= None:
            review=s_review.get_text()
            
            
        s_helpful = div.find('div', class_='text-muted')
        if s_helpful!=None:
            helpful=s_helpful.get_text().split('\n')[1].strip()


        reviews.loc[idx]=(title,reviewer,date,rating,review,helpful)
        
    
    return reviews


# In[5]:


# Test the function

executable_path = 'D:/Web Mining/Class Files/driver/chromedriver.exe'

driver = webdriver.Chrome(executable_path=executable_path)

page_url = 'https://www.imdb.com/title/tt1745960/reviews?sort=totalVotes&dir=desc&ratingFilter=0'
reviews = getReviews(page_url, driver)

driver.quit()

print(len(reviews))
print(reviews.head())


# In[ ]:




