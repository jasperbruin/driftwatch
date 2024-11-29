# LOCUSTFILE.PY with Bad Requests
import random
import pandas as pd
import numpy as np
from locust import HttpUser, TaskSet, between
from faker import Faker
import datetime
import re
import json

fake = Faker()
product_json = pd.read_json("products.json")
products = np.asarray(
    product_json["products"].apply(lambda x: x["id"]).astype(str))

with open("./product_id_mapper.json") as user_file:
    product_mapper = json.load(user_file)

# Probability of generating bad requests
BAD_REQUEST_PROB = 0.2  # 20% chance for any request to be malformed


def is_bad_request():
    """Determine whether to generate a bad request."""
    return random.random() < BAD_REQUEST_PROB


def index(l):
    l.client.get("/")


def setCurrency(l):
    currencies = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']
    if is_bad_request():
        # Send an invalid currency code
        l.client.post("/setCurrency", {'currency_code': 'INVALID_CODE'})
    else:
        l.client.post("/setCurrency",
                      {'currency_code': random.choice(currencies)})


def browseProduct(l):
    product_id = random.choice(products) if not is_bad_request() else "INVALID_ID"
    product_page = l.client.get(url=f"/product/{product_id}",
                                headers={"recommendation": "false"}).text
    if not is_bad_request():
        recommendations = []
        shown_products = re.findall("a href=\"/product/(.+)\"", product_page)
        for i in range(4):
            recommendations.append(shown_products[i])

        # Check if recommendations are valid before proceeding
        if recommendations[0] in product_mapper:
            ctr_prediction = np.mean([random.random()])  # Mock prediction
            getsTaken = random.choices([0, 1], weights=[1 - ctr_prediction, ctr_prediction])[0]
            if getsTaken == 1:
                browseRecommendation(l, random.choice(recommendations))
            else:
                addToCart(l)


def browseRecommendation(l, rec):
    l.client.get(url="/product/" + rec, headers={"recommendation": "true"})


def viewCart(l):
    l.client.get("/cart")


def addToCart(l):
    if is_bad_request():
        # Invalid product ID or quantity
        l.client.post("/cart", {'product_id': 'INVALID_ID', 'quantity': -5})
    else:
        product = random.choice(products)
        l.client.get(url="/product/" + product,
                     headers={"recommendation": "false"})
        l.client.post("/cart", {
            'product_id': product,
            'quantity': random.randint(1, 10)})


def checkout(l):
    if is_bad_request():
        # Send a malformed checkout request
        l.client.post("/cart/checkout", {
            'email': 'INVALID_EMAIL',  # Invalid email
            'street_address': '',  # Missing address
            'zip_code': 'INVALID_ZIP',
            'city': fake.city(),
            'state': fake.state_abbr(),
            'country': fake.country(),
            'credit_card_number': '12345',  # Invalid credit card
            'credit_card_expiration_month': 0,  # Invalid month
            'credit_card_expiration_year': 1900,  # Expired year
            'credit_card_cvv': '12',  # Invalid CVV
        })
    else:
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
            'credit_card_expiration_year': random.randint(current_year,
                                                          current_year + 70),
            'credit_card_cvv': f"{random.randint(100, 999)}",
        })


def logout(l):
    l.client.get('/logout')


# Dedicated bad request function
def generate_bad_requests(l):
    # Send a random invalid endpoint or request
    endpoints = ["/invalid_endpoint", "/cart", "/product"]
    l.client.get(random.choice(endpoints), headers={"invalid_header": "true"})


class UserBehavior(TaskSet):

    def on_start(self):
        index(self)

    tasks = {
        browseProduct: 10,
        addToCart: 2,
        checkout: 1,
        setCurrency: 1,
        generate_bad_requests: 1,  # Inject dedicated bad requests
    }


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 10)
