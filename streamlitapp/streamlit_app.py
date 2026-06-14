import os
import streamlit as st
import requests as re

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))

st.title("Car Price Prediction Web App")

st.write("""
## About

**This Streamlit App utilizes a Machine Learning model served as an API to predict the price of a car based on certain features.**

""")


st.header('Input Car Details')

names = st.text_input("""Name of Car""")
curbweight= st.number_input("""Input Curb Weight""")
enginesize = st.number_input("""Input Engine Size""")
horsepower = st.number_input("""Input Horse Power""")
highwaympg= st.number_input("""Input Highway Miles Per Gallon""")
carwidth = st.number_input("""Input Car Width""")
wheelbase = st.number_input("""Input Wheel Base""")
drivewheel = st.text_input("""Input Drive Wheel (fwd / rwd / 4wd)""")
citympg = st.number_input("""Input City Miles Per Gallon""")
boreratio = st.number_input("""Input Bore Ratio""")
cylindernumber = st.text_input("""Input Cylinder Number (e.g. four, six, eight)""")




if st.button("Predict Price"):
    if not names:
        st.error("Please enter the name of the car.")
    else:
        values = {
            "enginesize": enginesize,
            "curbweight": curbweight,
            "horsepower": horsepower,
            "highwaympg": highwaympg,
            "carwidth": carwidth,
            "wheelbase": wheelbase,
            "drivewheel": drivewheel,
            "citympg": citympg,
            "boreratio": boreratio,
            "cylindernumber": cylindernumber
        }

        url = f"{API_BASE_URL.rstrip('/')}/predict"
        try:
            res = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            body = res.json()
            prediction = body.get("prediction")
            if prediction is None:
                st.error(f"Unexpected response format from server: {body}")
            else:
                st.success(f"The Price of the {names} is {prediction:.2f}$")
        except re.exceptions.ConnectionError:
            st.error("Unable to connect to the prediction service. Please check that the API server is running.")
        except re.exceptions.Timeout:
            st.error("The request to the prediction service timed out. Please try again later.")
        except re.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = res.json().get("detail", "")
            except Exception:
                pass
            st.error(f"Server returned an error ({res.status_code}): {detail or str(e)}")
        except ValueError:
            st.error("The server returned an invalid response. Please try again later.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

