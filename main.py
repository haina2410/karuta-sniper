import random
from time import time
import discord
from dotenv import load_dotenv
import os
import asyncio
from utils import auto_message_task, handle_reaction

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = 1119245745212624926  # Use env var with fallback
KARUTA_ID = 646937666251915264

if not TOKEN:
    raise Exception("No token found in environment variables")


class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_react_time = 0

    async def on_ready(self):
        print("Logged on as", self.user)
        # Start the auto message task
        asyncio.create_task(auto_message_task(self, CHANNEL_ID))

    async def on_message(self, message):
        # only respond to Karuta
        if message.author.id != KARUTA_ID:
            return

        # Check if we've reacted recently (2 minutes cooldown)
        current_time = time()
        if (
            self.last_react_time > 0 and current_time - self.last_react_time < 60 * 2
        ):  # 2 minutes
            return

        if "dropping 3 cards" in message.content:
            # wait 0.9 to 2 seconds before reacting
            await asyncio.sleep(random.uniform(0.9, 2))

            reaction_time = await handle_reaction(message)
            if reaction_time:  # Only update if reaction was successful
                self.last_react_time = reaction_time

                await asyncio.sleep(random.uniform(0.9, 2))
                channel = self.get_channel(CHANNEL_ID)
                if channel:
                    await channel.send("kt a")
                else:
                    print(f"Channel with ID {CHANNEL_ID} not found")


client = MyClient()
client.run(TOKEN)
