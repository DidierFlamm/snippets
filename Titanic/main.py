import os
from streamlit.web import cli

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    cli.main_run([os.path.join(dir_path, "streamlit_app.py")])
