import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__),'.flaskenv')
load_dotenv(dotenv_path)

SECRET_KEY = os.getenv('SECRET_KEY')
from app import app