import json
import os
import shutil

import uvicorn
from fastapi import FastAPI, Depends, UploadFile, File, Security, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader, APIKey
from pydantic import BaseModel
from receipt_parser_core.config import read_config
from receipt_parser_core import Receipt
from pytesseract import pytesseract
import cv2
from starlette.responses import RedirectResponse
from starlette.status import HTTP_403_FORBIDDEN
from werkzeug.utils import secure_filename

import receipt_printer as printer
import util as util
from colors import bcolors

COOKIE_DOMAIN = "receipt.parser.de"
ALLOWED_PORT = int(os.environ.get('PORT', 8080))
ALLOWED_HOST = "0.0.0.0"

UPLOAD_FOLDER = 'data/img'
TMP_FOLDER = 'data/tmp/'
TRAINING_FOLDER = 'data/training/'
CERT_LOCATION = "cert/server.crt"
KEY_LOCATION = "cert/server.key"
DATA_PREFIX = "data/img/"
API_TOKEN_FILE = "data/.api_token"

# ZERO_CONF
ZERO_CONF_DESCRIPTION = "Receipt parser server._receipt-service._tcp.local."
ZERO_CONF_SERVICE = "_receipt-service._tcp.local."

API_KEY_NAME = "access_token"
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)

config = read_config(util.get_config_dir() + "config.yml")

if os.path.isfile(API_TOKEN_FILE):
    with open(API_TOKEN_FILE) as f:
        line = f.readline().strip()
        if not line:
           raise RuntimeError("can't find valid API token")
        else:
            API_KEY = line

else:
    raise RuntimeError("API token does not exist.")

class Receipt(BaseModel):
    company: str
    date: str
    total: str

async def get_api_key(
        api_query: str = Security(api_key_query),
        api_header: str = Security(api_key_header),
        api_cookie: str = Security(api_key_cookie),
):
    if api_query == API_KEY:
        return api_query
    elif api_header == API_KEY:
        return api_header
    elif api_cookie == API_KEY:
        return api_cookie
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )

# Set header and cookies
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
origins = [
    "https://receipt-parser.com",
    "https://receipt-parser.com:8721",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prepare training dataset for neuronal parser
# If an photo is submitted, upload the corresponding json file
@app.post("/api/training", tags=["api"])
async def get_open_api_endpoint(receipt: Receipt,
                                api_key: APIKey = Depends(get_api_key)):
    if not receipt:
        raise HTTPException(
            status_code=415, detail="Invalid receipt send"
        )

    search_dir = util.get_work_dir() + TMP_FOLDER
    file = util.get_last_modified_file(search_dir)

    search_dir = util.get_work_dir() + TRAINING_FOLDER
    last = util.get_last_modified_file(search_dir)

    index = 0
    if last:
        filename = os.path.basename(last).split(".")[0]
        if filename and filename == '':
            index = int(filename) + 1

    shutil.copyfile(file, util.get_work_dir() + TRAINING_FOLDER + str(index) + ".png")
    training_set = {'company': receipt.company, "date": receipt.date, "total": receipt.total}

    with open(TRAINING_FOLDER + str(index) + '.json', 'w+') as out:
        json.dump(training_set, out)

    return JSONResponse(content="success")


# Current image api
@app.post("/api/upload", tags=["api"])
async def get_open_api_endpoint(
        grayscale_image: bool = False,
        gaussian_blur: bool = False,
        rotate_image: bool = False,
        file: UploadFile = File(...),
        api_key: APIKey = Depends(get_api_key)):
    if file.filename == "":
        printer.error("No filename exist")
        raise HTTPException(
            status_code=415, detail="Invalid image send"
        )

    if file and util.allowed_file(file.filename):
        print(file.filename)

        filename = secure_filename(file.filename)
        output = os.path.join(util.get_work_dir() + UPLOAD_FOLDER, filename)
        printer.info("Store file at: " + output)

        with open(output, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        printer.info("Parsing image")
        process_receipt(config, filename, rotate=rotate_image, grayscale=grayscale_image, gaussian_blur=gaussian_blur)

    else:
        raise HTTPException(
            status_code=415, detail="Invalid image send"
        )


@app.get("/logout")
async def route_logout_and_remove_cookie():
    response = RedirectResponse(url="/")
    response.delete_cookie(API_KEY_NAME, domain=COOKIE_DOMAIN)
    return response

if __name__ == "__main__":
    print("Current API token: " + bcolors.OKGREEN + API_KEY + bcolors.ENDC)
    print()

    if config.https:
        uvicorn.run("receipt_server:app", host="0.0.0.0", port=ALLOWED_PORT, log_level="info",
                    ssl_certfile=util.get_work_dir() + CERT_LOCATION, ssl_keyfile=util.get_work_dir() + KEY_LOCATION)
    else:
        uvicorn.run("receipt_server:app", host="0.0.0.0", port=ALLOWED_PORT, log_level="info")

BASE_PATH = os.getcwd()
INPUT_FOLDER = os.path.join(BASE_PATH, "data/img")
TMP_FOLDER = os.path.join(BASE_PATH, "data/tmp")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "data/txt")

ORANGE = '\033[33m'
RESET = '\033[0m'

def prepare_folders():
    """
    :return: void
        Creates necessary folders
    """

    for folder in [
        INPUT_FOLDER, TMP_FOLDER, OUTPUT_FOLDER
    ]:
        if not os.path.exists(folder):
            os.makedirs(folder)

def process_receipt(config, filename, rotate=True, grayscale=True, gaussian_blur=True):
    input_path = INPUT_FOLDER + "/" + filename

    output_path = OUTPUT_FOLDER + "/" + filename.split(".")[0] + ".txt"

    print(ORANGE + '~: ' + RESET + 'Process image (Rob): ' + ORANGE + input_path + RESET)
    prepare_folders()

    try:
        img = cv2.imread(input_path)
    except FileNotFoundError:
        return Receipt(config=config, raw="")

    tmp_path = os.path.join(
        TMP_FOLDER, filename
    )

    print(ORANGE + '~: ' + RESET + 'Temporary store image at  (Rob): ' + ORANGE + tmp_path + RESET)

    cv2.imwrite(tmp_path, img)
    run_tesseract(tmp_path, output_path)

    print(ORANGE + '~: ' + RESET + 'Store parsed text at  (Rob): ' + ORANGE + output_path + RESET)
    raw = open(output_path, 'r').readlines()
    print(raw)

def run_tesseract(input_file, output_file):
    """
    :param input_file: str
        Path to image to OCR
    :param output_file: str
        Path to output file
    :return: void
        Runs tesseract on image and saves result
    """

    print(ORANGE + '\t~: ' + RESET + 'Parse image using pytesseract' + RESET)
    print(ORANGE + '\t~: ' + RESET + 'Parse image at: ' + input_file + RESET)
    print(ORANGE + '\t~: ' + RESET + 'Write result to: ' + output_file + RESET)

    image_data = pytesseract.image_to_string(input_file, timeout=60, config="--psm 6")

    out = open(output_file, "w", encoding='utf-8')
    out.write(image_data)
    out.close()
    return image_data