import asyncio
import random
import logging
from time import time, strftime, localtime

logger = logging.getLogger(__name__)

MIN_SLEEP_INTERVAL = 30 * 60  # 30 minutes in seconds
MAX_SLEEP_INTERVAL = 33 * 60  # 33 minutes in seconds


async def auto_message_task(client, channel_id):
    """Background task that sends 'kd' message every 30 minutes"""
    await client.wait_until_ready()

    # Send first message immediately
    try:
        channel = client.get_channel(channel_id)
        if channel:
            await channel.send(f"kd <@{411465198713700353}>")
            logger.info(
                f"Auto message 'kd' sent to {channel.name} at {strftime('%Y-%m-%d %H:%M:%S', localtime(time()))}"
            )
        else:
            logger.error(f"Channel with ID {channel_id} not found")
    except Exception as e:
        logger.error(f"Error sending initial auto message: {e}")

    while not client.is_closed():
        sleep_time = random.randint(MIN_SLEEP_INTERVAL, MAX_SLEEP_INTERVAL)
        logger.info(
            f"Waiting to send auto message. Next in: {strftime('%Y-%m-%d %H:%M:%S', localtime(time() + sleep_time))}"
        )
        await asyncio.sleep(sleep_time)
        try:
            channel = client.get_channel(channel_id)
            if channel:
                await channel.send("kd")
                logger.info(
                    f"Auto message 'kd' sent to {channel.name} at {strftime('%Y-%m-%d %H:%M:%S', localtime(time()))}"
                )
            else:
                logger.error(f"Channel with ID {channel_id} not found")
        except Exception as e:
            logger.error(f"Error sending auto message: {e}")


REACTIONS = ["1️⃣", "2️⃣", "3️⃣"]


async def handle_reaction(message):
    """Handle reactions to messages"""
    # Example: Add a reaction to the message
    try:
        await message.add_reaction(random.choice(REACTIONS))
        return time()  # Return current timestamp on success
    except Exception as e:
        logger.error(f"Error adding reaction: {e}")
        return None  # Return None on error
