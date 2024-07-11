import asyncio
import logging
import sys
from aiogram.types import FSInputFile
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.keyboard import InlineKeyboardButton
import aiohttp
from db_bot import init_db, save_to_db

from PIL import Image
from dotenv import load_dotenv
import os
import io
import pandas as pd

load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")

WEB_APP_URL = os.environ.get("WEB_APP_URL")

SERVER_URL = 'http://127.0.0.1:8000/upload/'#поменяй для локальных тестов

# Configure logging
logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

init_db()

@dp.message(Command(commands=['start']))
async def register_handler(message) -> None:

    await message.answer("Отправьте фото схемы для обработки в чат.")


@dp.message()
async def handle_docs_photo(message: types.Message):
    file_id = None
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document and message.document.mime_type.startswith('image/'):
        file_id = message.document.file_id

    if file_id:
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path

        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_path}') as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    image = Image.open(io.BytesIO(image_data))
                    png_image_data = io.BytesIO()
                    image.save(png_image_data, format='PNG')
                    png_image_data.seek(0)
                    # Here you would send the data to your microservice
                    # async with session.post('http://localhost:8000/compile', data=png_image_data) as resp:
                    #     if resp.status == 200:
                    # model-generated xlsx file should be received instead of the placeholder xslx below:
                    if not os.path.exists('output'):
                        os.makedirs('output')
                    data = {'Артикул': ['1', '2', '3','4','5','6'], 'Номенклатура': ['шкаф', 'скуф', 'скуф','шкаф','шкаф','скуф'], 'Количество': [1, 10, 2,1,1,1], 'Цена': [100, 10, 50,5,5,1], 'Стоимость': [100, 100, 100,5,5,1]}
                    df = pd.DataFrame(data)
                    output = io.BytesIO()

                    # Use the XlsxWriter Excel writer
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)

                    # Seek to the beginning of the stream
                    output.seek(0)

                    # Get the Excel data as bytes
                    excel_data = output.getvalue()

                    with open(f'output/{file_id}.xlsx', 'wb') as f:
                        f.write(excel_data)
                    # Save to database
                    save_to_db(message.from_user.id, file_id)
                    # Send the excel file
                    data_post = aiohttp.FormData()
                    with open(f'output/{file_id}.xlsx', 'rb') as f:
                        data_post.add_field('file', f, filename=f'{file_id}')
                        async with session.post(SERVER_URL, data=data_post) as post_response:
                            if post_response.status == 200:
                                print()
                    inline_webapp = InlineKeyboardBuilder()
                    web_app_url = f'{WEB_APP_URL}/?file_id={file_id}'
                    inline_webapp.add(InlineKeyboardButton(text="Открыть",
                                                           web_app=types.WebAppInfo(url=web_app_url)))

                    await message.answer_document(FSInputFile(f'output/{file_id}.xlsx'), reply_markup=inline_webapp.as_markup())


async def main() -> None:
    bot = Bot(token=TOKEN)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
