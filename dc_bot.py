import discord
import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL")
ADMIN_ID = int(os.getenv("ADMIN_ID"))

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    try:
        response = requests.post(API_URL, json={"text": message.content})
        result = response.json()

        if result["pred_label"] == 1:
            await message.delete()

            await message.channel.send(
                f"{message.author.mention}'s message has been deleted due to toxicity."
            )

            user = await client.fetch_user(ADMIN_ID)

            await user.send(
                f"Cyberbullying detected!\n\n"
                f"User: {message.author}\n"
                f"Message: {message.content}\n"
                f"Confidence: {result['prob_cyberbullying']:.4f}"
            )

    except Exception as e:
        print("Error:", e)

client.run(TOKEN)