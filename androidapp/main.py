import os
import json
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivy.storage.jsonstore import JsonStore
import certifi as cfi
import requests as re

DEFAULT_API_URL = "http://10.0.2.2:8000"
DEFAULT_TIMEOUT = 15

KV = """
ScreenManager:
    MainScreen:
    SettingsScreen:

<MainScreen>:
    name: 'main'

    MDLabel:
        text: 'Car Price Prediction'
        halign: 'center'
        pos_hint: {'center_y': 0.95}
        font_style: 'H4'

    MDLabel:
        text: 'Engine Size'
        pos_hint: {'center_y': 0.86, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_enginesize
        hint_text: 'e.g., 130'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.86, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Curb Weight'
        pos_hint: {'center_y': 0.80, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_curbweight
        hint_text: 'e.g., 2548'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.80, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Horsepower'
        pos_hint: {'center_y': 0.74, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_horsepower
        hint_text: 'e.g., 111'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.74, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Highway MPG'
        pos_hint: {'center_y': 0.68, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_highwaympg
        hint_text: 'e.g., 27'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.68, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Car Width'
        pos_hint: {'center_y': 0.62, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_carwidth
        hint_text: 'e.g., 64.1'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.62, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Wheel Base'
        pos_hint: {'center_y': 0.56, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_wheelbase
        hint_text: 'e.g., 88.6'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.56, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Drive Wheel'
        pos_hint: {'center_y': 0.50, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_drivewheel
        hint_text: 'rwd, fwd, or 4wd'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.50, 'center_x': 0.5}

    MDLabel:
        text: 'City MPG'
        pos_hint: {'center_y': 0.44, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_citympg
        hint_text: 'e.g., 21'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.44, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Bore Ratio'
        pos_hint: {'center_y': 0.38, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_boreratio
        hint_text: 'e.g., 3.47'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.38, 'center_x': 0.5}
        input_filter: 'float'

    MDLabel:
        text: 'Cylinders'
        pos_hint: {'center_y': 0.32, 'center_x': 0.55}
        font_style: 'Caption'

    MDTextField:
        id: input_cylindernumber
        hint_text: 'four, six, etc.'
        width: 120
        size_hint_x: None
        pos_hint: {'center_y': 0.32, 'center_x': 0.5}

    MDLabel:
        id: output_text
        pos_hint: {'center_y': 0.22}
        halign: 'center'
        text: ''
        theme_text_color: "Custom"
        text_color: 0, 0.8, 0, 1
        size_hint_x: 0.9
        text_size: self.width, None

    MDRaisedButton:
        pos_hint: {'center_x': 0.3, 'center_y': 0.12}
        text: 'Predict'
        on_press: app.predict()
        md_bg_color: app.theme_cls.primary_color

    MDRaisedButton:
        pos_hint: {'center_x': 0.7, 'center_y': 0.12}
        text: 'Settings'
        on_press: app.go_to_settings()
        md_bg_color: 0.5, 0.5, 0.5, 1

    MDLabel:
        id: connection_status
        pos_hint: {'center_y': 0.05}
        halign: 'center'
        text: ''
        font_style: 'Caption'
        theme_text_color: "Custom"
        text_color: 0.5, 0.5, 0.5, 1

<SettingsScreen>:
    name: 'settings'

    MDLabel:
        text: 'API Configuration'
        halign: 'center'
        pos_hint: {'center_y': 0.9}
        font_style: 'H4'

    MDLabel:
        text: 'API Base URL'
        pos_hint: {'center_y': 0.78, 'center_x': 0.3}
        font_style: 'Caption'

    MDTextField:
        id: api_url_input
        hint_text: 'http://your-api-server:8000'
        width: 250
        size_hint_x: None
        pos_hint: {'center_y': 0.78, 'center_x': 0.6}

    MDLabel:
        text: 'Request Timeout (sec)'
        pos_hint: {'center_y': 0.70, 'center_x': 0.3}
        font_style: 'Caption'

    MDTextField:
        id: timeout_input
        hint_text: '15'
        width: 100
        size_hint_x: None
        pos_hint: {'center_y': 0.70, 'center_x': 0.6}
        input_filter: 'int'

    MDLabel:
        id: health_status
        pos_hint: {'center_y': 0.60}
        halign: 'center'
        text: ''
        font_style: 'Caption'
        theme_text_color: "Custom"
        text_color: 0.5, 0.5, 0.5, 1

    MDRaisedButton:
        pos_hint: {'center_x': 0.3, 'center_y': 0.50}
        text: 'Test Connection'
        on_press: app.test_connection()

    MDRaisedButton:
        pos_hint: {'center_x': 0.7, 'center_y': 0.50}
        text: 'Save & Return'
        on_press: app.save_settings()
        md_bg_color: app.theme_cls.primary_color

    MDLabel:
        text: 'Note: For Android emulator use 10.0.2.2 to access localhost'
        pos_hint: {'center_y': 0.35}
        halign: 'center'
        font_style: 'Caption'
        theme_text_color: "Custom"
        text_color: 0.6, 0.6, 0.6, 1

    MDLabel:
        id: current_config
        pos_hint: {'center_y': 0.25}
        halign: 'center'
        text: ''
        font_style: 'Caption'
        theme_text_color: "Custom"
        text_color: 0.4, 0.4, 0.4, 1
"""


class MainScreen(Screen):
    pass


class SettingsScreen(Screen):
    pass


class MainApp(MDApp):
    def build(self):
        self.store = JsonStore("app_config.json")
        self.api_base_url = self.store.get("api_config", {}).get(
            "base_url", os.environ.get("API_BASE_URL", DEFAULT_API_URL)
        )
        self.request_timeout = self.store.get("api_config", {}).get(
            "timeout", int(os.environ.get("API_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT)))
        )

        self.sm = Builder.load_string(KV)
        Clock.schedule_once(self.update_connection_status, 0.5)
        return self.sm

    def update_connection_status(self, *args):
        main_screen = self.sm.get_screen("main")
        main_screen.ids.connection_status.text = f"API: {self.api_base_url}"

    def go_to_settings(self):
        settings_screen = self.sm.get_screen("settings")
        settings_screen.ids.api_url_input.text = self.api_base_url
        settings_screen.ids.timeout_input.text = str(self.request_timeout)
        settings_screen.ids.current_config.text = f"Current: {self.api_base_url} (Timeout: {self.request_timeout}s)"
        self.sm.current = "settings"

    def save_settings(self):
        settings_screen = self.sm.get_screen("settings")
        new_url = settings_screen.ids.api_url_input.text.strip()
        new_timeout_str = settings_screen.ids.timeout_input.text.strip()

        if not new_url:
            settings_screen.ids.health_status.text = "Error: API URL cannot be empty"
            settings_screen.ids.health_status.text_color = 1, 0, 0, 1
            return

        try:
            new_timeout = int(new_timeout_str) if new_timeout_str else DEFAULT_TIMEOUT
            if new_timeout < 1 or new_timeout > 120:
                raise ValueError("Timeout must be between 1 and 120 seconds")
        except ValueError as e:
            settings_screen.ids.health_status.text = f"Error: {str(e)}"
            settings_screen.ids.health_status.text_color = 1, 0, 0, 1
            return

        self.api_base_url = new_url.rstrip("/")
        self.request_timeout = new_timeout
        self.store.put("api_config", base_url=self.api_base_url, timeout=self.request_timeout)

        main_screen = self.sm.get_screen("main")
        main_screen.ids.connection_status.text = f"API: {self.api_base_url}"

        settings_screen.ids.health_status.text = "Settings saved successfully!"
        settings_screen.ids.health_status.text_color = 0, 0.8, 0, 1
        Clock.schedule_once(lambda dt: self.sm.switch_to(self.sm.get_screen("main")), 1)

    def test_connection(self):
        settings_screen = self.sm.get_screen("settings")
        test_url = settings_screen.ids.api_url_input.text.strip().rstrip("/")

        if not test_url:
            settings_screen.ids.health_status.text = "Error: Please enter an API URL"
            settings_screen.ids.health_status.text_color = 1, 0, 0, 1
            return

        settings_screen.ids.health_status.text = "Testing connection..."
        settings_screen.ids.health_status.text_color = 0.5, 0.5, 0.5, 1

        def do_test(dt):
            try:
                health_url = f"{test_url}/health"
                resp = re.get(health_url, timeout=self.request_timeout, verify=cfi.where())
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") == "ok":
                    version = data.get("version", "N/A")
                    model_loaded = "Yes" if data.get("model_loaded") else "No"
                    settings_screen.ids.health_status.text = f"✓ Connected! Version: {version}, Model: {model_loaded}"
                    settings_screen.ids.health_status.text_color = 0, 0.8, 0, 1
                else:
                    settings_screen.ids.health_status.text = f"✗ Service not ready: {data}"
                    settings_screen.ids.health_status.text_color = 1, 0.5, 0, 1
            except re.exceptions.ConnectionError:
                settings_screen.ids.health_status.text = "✗ Cannot connect to server"
                settings_screen.ids.health_status.text_color = 1, 0, 0, 1
            except re.exceptions.Timeout:
                settings_screen.ids.health_status.text = "✗ Connection timed out"
                settings_screen.ids.health_status.text_color = 1, 0.5, 0, 1
            except Exception as e:
                settings_screen.ids.health_status.text = f"✗ Error: {str(e)}"
                settings_screen.ids.health_status.text_color = 1, 0, 0, 1

        Clock.schedule_once(do_test, 0.1)

    def predict(self):
        main_screen = self.sm.get_screen("main")
        output = main_screen.ids.output_text
        output.text = "Predicting..."
        output.text_color = 0.5, 0.5, 0.5, 1

        def do_predict(dt):
            try:
                values = {
                    "enginesize": float(main_screen.ids.input_enginesize.text),
                    "curbweight": float(main_screen.ids.input_curbweight.text),
                    "horsepower": float(main_screen.ids.input_horsepower.text),
                    "highwaympg": float(main_screen.ids.input_highwaympg.text),
                    "carwidth": float(main_screen.ids.input_carwidth.text),
                    "wheelbase": float(main_screen.ids.input_wheelbase.text),
                    "drivewheel": main_screen.ids.input_drivewheel.text.strip(),
                    "citympg": float(main_screen.ids.input_citympg.text),
                    "boreratio": float(main_screen.ids.input_boreratio.text),
                    "cylindernumber": main_screen.ids.input_cylindernumber.text.strip(),
                }

                url = f"{self.api_base_url}/predict"
                resp = re.post(url=url, json=values, timeout=self.request_timeout, verify=cfi.where())
                resp.raise_for_status()
                body = resp.json()

                if body.get("status") == "error":
                    error_type = body.get("error", "UnknownError")
                    detail = body.get("detail", "No details available")
                    output.text = f"Error [{error_type}]: {detail}"
                    output.text_color = 1, 0, 0, 1
                    return

                prediction = body.get("prediction")
                if prediction is None:
                    output.text = "Error: Server response missing 'prediction' field"
                    output.text_color = 1, 0, 0, 1
                else:
                    output.text = f"Predicted Price: ${prediction:.2f}"
                    output.text_color = 0, 0.8, 0, 1

            except ValueError as e:
                output.text = f"Input Error: Please check all numeric fields"
                output.text_color = 1, 0.5, 0, 1
            except re.exceptions.ConnectionError:
                output.text = "Connection Error: Cannot reach the prediction service. Check API URL and network."
                output.text_color = 1, 0, 0, 1
            except re.exceptions.Timeout:
                output.text = "Timeout: Request took too long. Try increasing timeout in Settings."
                output.text_color = 1, 0.5, 0, 1
            except re.exceptions.HTTPError as e:
                detail = ""
                error_type = ""
                try:
                    err_data = resp.json()
                    detail = err_data.get("detail", "")
                    error_type = err_data.get("error", "")
                except Exception:
                    pass

                if resp.status_code == 400:
                    output.text = f"Invalid Input: {detail or 'Please check your values'}"
                    output.text_color = 1, 0.5, 0, 1
                elif resp.status_code == 404:
                    output.text = "Error: Prediction endpoint not found. Check API URL."
                    output.text_color = 1, 0, 0, 1
                elif resp.status_code == 500:
                    output.text = f"Server Error: {detail or 'Internal server error'}"
                    output.text_color = 1, 0, 0, 1
                else:
                    output.text = f"HTTP Error {resp.status_code}: {error_type or str(e)}"
                    output.text_color = 1, 0, 0, 1
            except Exception as e:
                output.text = f"Unexpected Error: {str(e)}"
                output.text_color = 1, 0, 0, 1

        Clock.schedule_once(do_predict, 0.1)


if __name__ == "__main__":
    MainApp().run()
