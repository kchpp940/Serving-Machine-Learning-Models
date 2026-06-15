import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
import certifi as cfi
import requests as re

from car_pricing.api_client import (
    create_client,
    ServiceError,
    ErrorCategory,
)

SERVICE_TYPE = os.environ.get("API_SERVICE_TYPE", "fastapi")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))


Builder_string = """
ScreenManager:
    Main:
<Main>:

    name : 'main'
    MDLabel:
        text: 'Car Price Prediction App'
        halign: 'center'
        pos_hint: {'center_y':0.9}
        font_style: 'H3'

    MDLabel:
        text: 'Engine Size'
        pos_hint: {'center_y':0.75, 'center_x':0.55}

    MDTextField:
        id: input_1
        hint_text: '(0.0 - 3.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.75, 'center_x':0.5}

    MDLabel:
        text: 'Curb Weight'
        pos_hint: {'center_y':0.68, 'center_x':0.55}

    MDTextField:
        id: input_2
        hint_text: '(0.0 - 3.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.68, 'center_x':0.5}

    MDLabel:
        text: 'Horsepower'
        pos_hint: {'center_y':0.61, 'center_x':0.55}

    MDTextField:
        id: input_3
        hint_text: '(0.0 - 3.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.61, 'center_x':0.5}

    MDLabel:
        text: 'Highway Miles Per Gallon'
        pos_hint: {'center_y':0.54, 'center_x':0.55}

    MDTextField:
        id: input_4
        hint_text: '(0.0 - 4)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.54, 'center_x':0.5}

    MDLabel:
        text: 'Car Width'
        pos_hint: {'center_y':0.47, 'center_x':0.55}

    MDTextField:
        id: input_5
        hint_text: '(0.0 - 20.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.47, 'center_x':0.5}

    MDLabel:
        text: 'Wheel Base'
        pos_hint: {'center_y':0.40, 'center_x':0.55}

    MDTextField:
        id: input_6
        hint_text: '(0.0 - 1.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.40, 'center_x':0.5}

    MDLabel:
        text: 'Drive Wheel'
        pos_hint: {'center_y':0.33, 'center_x':0.55}

    MDTextField:
        id: input_7
        hint_text: '(0.0 - 2.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.33, 'center_x':0.5}

    MDLabel:
        text: 'City MPG'
        pos_hint: {'center_y':0.26, 'center_x':0.55}

    MDTextField:
        id: input_8
        hint_text: '(0.0 - 3.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.26, 'center_x':0.5}
    
    MDLabel:
        text: 'Bore Ratio'
        pos_hint: {'center_y':0.20, 'center_x':0.55}

    MDTextField:
        id: input_9
        hint_text: '(0.0 - 3.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.20, 'center_x':0.5}

    MDLabel:
        text: 'Cylinder Number'
        pos_hint: {'center_y':0.14, 'center_x':0.55}

    MDTextField:
        id: input_10
        hint_text: '(0.0 - 3.0)'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y':0.16, 'center_x':0.5}


    MDLabel:
        pos_hint: {'center_y':0.2}
        halign: 'center'
        text: ''
        id: output_text
        theme_text_color: "Custom"
        text_color: 0, 1, 0, 1

    MDRaisedButton:
        pos_hint: {'center_x':0.5, 'center_y':0.1}
        text: 'Predict'
        on_press: app.predict()
"""


class Main(Screen):
    pass


sm = ScreenManager()
sm.add_widget(Main(name="main"))


class MainApp(MDApp):
    def build(self):
        self.help_string = Builder.load_string(Builder_string)
        self._client = create_client(
            service_type=SERVICE_TYPE,
            base_url=API_BASE_URL,
            timeout=REQUEST_TIMEOUT,
            verify_ssl=cfi.where(),
        )
        return self.help_string

    def predict(self):
        enginesize = self.help_string.get_screen("main").ids.input_1.text
        curbweight = self.help_string.get_screen("main").ids.input_2.text
        horsepower = self.help_string.get_screen("main").ids.input_3.text
        highwaympg = self.help_string.get_screen("main").ids.input_4.text
        carwidth = self.help_string.get_screen("main").ids.input_5.text
        wheelbase = self.help_string.get_screen("main").ids.input_6.text
        drivewheel = self.help_string.get_screen("main").ids.input_7.text
        citympg = self.help_string.get_screen("main").ids.input_8.text
        boreratio = self.help_string.get_screen("main").ids.input_9.text
        cylindernumber = self.help_string.get_screen("main").ids.input_10.text
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
        output = self.help_string.get_screen("main").ids.output_text
        try:
            body = self._client.predict(values)
            prediction = body.get("prediction")
            if prediction is None:
                output.text = f"Error: unexpected response from server"
            else:
                output.text = f"Predicted Price: {prediction:.2f}$"
        except ServiceError as e:
            if e.category == ErrorCategory.CONNECTION:
                output.text = "Error: cannot connect to prediction service"
            elif e.category == ErrorCategory.TIMEOUT:
                output.text = "Error: request timed out, please try again"
            elif e.category == ErrorCategory.SERVER_ERROR or e.category == ErrorCategory.BAD_REQUEST:
                detail = e.raw_detail or ""
                output.text = f"Server error: {detail or 'unknown error'}"
            else:
                output.text = f"Error: {str(e)}"
        except ValueError:
            output.text = "Error: invalid response from server"
        except Exception as e:
            output.text = f"Error: {str(e)}"

    def res(self, *args):
        pass


MainApp().run()
