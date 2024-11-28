#!/usr/bin/python
import os
import random
import pandas as pd
import numpy as np
from locust import HttpUser, TaskSet, between
from faker import Faker
import datetime
import re
import torch
import json

# Initialize Faker and data
fake = Faker()
product_json = pd.read_json("products.json")
products = np.asarray(product_json["products"].apply(lambda x: x["id"]).astype(str))
model = torch.load("./model_formatted_fashion_cpu_v2")
with open("./product_id_mapper.json") as user_file:
    product_mapper = json.load(user_file)

# Parse environment variable for drift generation
ENABLE_DRIFT = os.getenv('ENABLE_DRIFT', 'false').lower() == 'true'

def apply_drift(product_ids):
    """
    Applies drift to the given list of product IDs.
    This function simulates changes to product distributions.
    """
    if ENABLE_DRIFT:
        # Example: Shuffle product IDs to mimic drift
        drifted_products = random.sample(list(product_ids), len(product_ids))
        print("Drift applied: Product IDs shuffled.")
        return drifted_products
    return product_ids

# Apply drift to products if enabled
products = apply_drift(products)

# Define tasks
def index(l):
    l.client.get("/")

def setCurrency(l):
    currencies = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']
    l.client.post("/setCurrency",
        {'currency_code': random.choice(currencies)})

def browseProduct(l):
    product_page = l.client.get(url="/product/" + random.choice(products),
                                headers={"recommendation": "false"}).text
    recommendations = []
    shown_products = re.findall("a href=\"/product/(.+)\"", product_page)

    for i in range(4):
        recommendations.append(shown_products[i])

    if recommendations[0] not in product_mapper:
        print(f"KeyError: '{recommendations[0]}' not found in product_mapper")
        return  # Skip this task if the key is missing

    ctr_prediction = model.predict({
        "user_id": pd.Series([random.randrange(100)]),
        "parent_asin": pd.Series([product_mapper[recommendations[0]]]),
        "year": pd.Series([random.randrange(10)]),
        "month": pd.Series([random.randrange(12)]),
        "day": pd.Series([random.randrange(28)])
    })

    ctr = np.mean(ctr_prediction)
    getsTaken = random.choices([0, 1], weights=[1 - ctr, ctr])[0]
    if getsTaken == 1:
        browseRecommendation(l, random.choice(recommendations))
    else:
        addToCart(l)

def browseRecommendation(l, rec):
    l.client.get(url="/product/" + rec, headers={"recommendation": "true"})

def viewCart(l):
    l.client.get("/cart")

def addToCart(l):
    product = random.choice(products)
    l.client.get(url="/product/" + product,
                 headers={"recommendation": "false"})
    l.client.post("/cart", {
        'product_id': product,
        'quantity': random.randint(1, 10)})

def empty_cart(l):
    l.client.post('/cart/empty')

def checkout(l):
    addToCart(l)
    current_year = datetime.datetime.now().year + 1
    l.client.post("/cart/checkout", {
        'email': fake.email(),
        'street_address': fake.street_address(),
        'zip_code': fake.zipcode(),
        'city': fake.city(),
        'state': fake.state_abbr(),
        'country': fake.country(),
        'credit_card_number': fake.credit_card_number(card_type="visa"),
        'credit_card_expiration_month': random.randint(1, 12),
        'credit_card_expiration_year': random.randint(current_year, current_year + 70),
        'credit_card_cvv': f"{random.randint(100, 999)}",
    })

def logout(l):
    l.client.get('/logout')

class UserBehavior(TaskSet):
    def on_start(self):
        index(self)

    tasks = {browseProduct: 10, checkout: 1}

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 10)
