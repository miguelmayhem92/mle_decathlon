import datetime
import json

import streamlit as st
import requests

st.title("""
    # MLE usecase: test inferences here 
""")

st.write("""
    Lets predict
""")

st.write("""
    example values:
         
         * start date: 2017-01-01, 2017-10-01

         * business unit: 4, 64, 119

         * department number: 128, 73, 88
         
""")

url = "http://host.docker.internal:9000/2015-03-31/functions/function/invocations"

input_date = str(st.date_input("date id", value = datetime.date(2017, 11, 25)))
b_number = int(st.number_input("business unit", value = 93))
d_number = int(st.number_input("department number", value = 127))

if st.button('get prediction'):
    payload = {'day_id': [input_date],
    'but_num_business_unit': [b_number],
    'dpt_num_department': [d_number]}

    response = requests.post(url, json=payload)
    
    st.write(f"response status code: {response.status_code}",)
    st.write("raw response:")
    st.write(response.json())
