import datetime
import json

import streamlit as st
import requests

st.write("""
    Lets predict
""")

url = "http://localhost:9000/2015-03-31/functions/function/invocations"
url = "http://host.docker.internal:9000/2015-03-31/functions/function/invocations"

input_date = str(st.date_input("date id", datetime.date(2017, 11, 25)))
b_number = int(st.number_input("business unit", 93))
d_number = int(st.number_input("department number", 127))

if st.button('get prediction'):
    payload = {'day_id': [input_date],
    'but_num_business_unit': [b_number],
    'dpt_num_department': [d_number]}

    response = requests.post(url, json=payload)
    st.write(response.status_code)
    st.write(response.json())
