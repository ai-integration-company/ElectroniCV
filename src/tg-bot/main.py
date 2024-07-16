import asyncio
import logging
import sys
from aiogram.types import FSInputFile, InputFile, BufferedInputFile
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.keyboard import InlineKeyboardButton
import aiohttp
import db
from pdf2image import convert_from_path

from PIL import Image
import base64
from dotenv import load_dotenv
import os
import io
import pandas as pd

load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")

WEB_APP_URL = os.environ.get("WEB_APP_URL")

ML_URL = os.environ.get("ML_URL")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

db.init_db()


@dp.message(Command(commands=['start']))
async def register_handler(message) -> None:

    await message.answer("Отправьте фото схемы для обработки в чат.")


@dp.message()
async def handle_docs_photo(message: types.Message):
    file_id = None
    is_pdf = False
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document and message.document.mime_type.startswith('image/'):
        file_id = message.document.file_id
    elif message.document and message.document.mime_type.startswith('application/pdf'):
        file_id = message.document.file_id
        is_pdf = True

    if file_id:
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path

        timeout = aiohttp.ClientTimeout(total=6000)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_path}') as resp:
                if resp.status == 200:
                    if is_pdf:
                        image_data_set = convert_from_path(f'https://api.telegram.org/file/bot{TOKEN}/{file_path}')
                        image = image_data_set[0]
                    else:
                        image_data = await resp.read()
                        image = Image.open(io.BytesIO(image_data))

                    png_image_data = io.BytesIO()
                    image.save(png_image_data, format='PNG')
                    encoded_image = base64.b64encode(png_image_data.getvalue())
                    # with open('file.txt','w') as f:
                    #    f.write(encoded_image.decode('utf-8'))
                    #    await message.answer_document(FSInputFile('file.txt'))

                    # Sending the data to the microservice
                    data = {'article': [], 'name': [], 'amount': [], 'price': [], 'sum': []}
                    async with session.post(ML_URL, json={'image': encoded_image.decode()}) as ml_resp:
                        if ml_resp.status == 200:
                            data = await ml_resp.json()

                    df = pd.DataFrame(data)
                    df_rus = pd.DataFrame(data)
                    df_rus.rename(columns=db.column_mapping, inplace=True)
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_rus.to_excel(writer, index=False)
                    output.seek(0)
                    excel_data = output.getvalue()

                    # Save to database
                    conn = db.get_db()
                    db.create_file_data(conn, message.from_user.id, file_id, df.to_dict(orient='records'))
                    conn.close()
                    # Send the excel file
                    inline_webapp = InlineKeyboardBuilder()
                    if WEB_APP_URL == "https://adam-suliman.github.io/EKF_web/":
                        web_app_url = f'{WEB_APP_URL}/?user_id={message.from_user.id}&file_id={file_id}'
                    else:
                        web_app_url = f'{WEB_APP_URL}'
                    inline_webapp.add(InlineKeyboardButton(text="Открыть",
                                                           web_app=types.WebAppInfo(url=web_app_url)))

                    await message.answer_document(BufferedInputFile(excel_data, f'{file_id}.xlsx'), reply_markup=inline_webapp.as_markup())


async def main() -> None:
    bot = Bot(token=TOKEN)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
