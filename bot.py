import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardMarkup, ReplyKeyboardBuilder, InlineKeyboardButton, KeyboardButton
import aiohttp
import sqlite3
import websockets

from PIL import Image
from dotenv import load_dotenv
import os
import io

load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")

WEB_APP_URL = os.environ.get("WEB_APP_URL")

# Configure logging
logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

inline_webapp = InlineKeyboardBuilder()

def init_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, file_id TEXT, excel_file TEXT)''')
    conn.commit()
    conn.close()
    
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
                    async with session.post('http://localhost:8000/compile', data=png_image_data) as resp:
                        if resp.status == 200:
                            excel_data = await resp.read()
                            # Save excel data to file
                            with open('output.xlsx', 'wb') as f:
                                f.write(excel_data)
                            # Save to database
                            conn = sqlite3.connect('data.db')
                            cursor = conn.cursor()
                            cursor.execute('INSERT INTO files (user_id, excel_file) VALUES (?, ?)', (message.from_user.id, message.photo[-1].file_id))
                            conn.commit()
                            conn.close()
                            # Send the excel file
                            await message.reply_document(types.InputFile(message.photo[-1].file_id))
                            # Send inline button
                            markup = InlineKeyboardMarkup()
                            web_app_url = f'index.html/?file_id={file_id}'
                            markup.add(InlineKeyboardButton("Open in Mini App", web_app=types.WebAppInfo(url=web_app_url)))
                            await message.reply("Click the button below to view the info in the mini app", reply_markup=markup)



async def main() -> None:
    bot = Bot(token=TOKEN)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
