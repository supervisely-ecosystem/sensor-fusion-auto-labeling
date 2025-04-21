import supervisely as sly
from dotenv import load_dotenv


# load credentials
load_dotenv("local.env")
load_dotenv("supervisely.env")
api = sly.Api()
