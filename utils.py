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

    while not client.is_closed():
        sleep_time = random.randint(MIN_SLEEP_INTERVAL, MAX_SLEEP_INTERVAL)
        logger.info(
            f"Waiting to send auto message. Next in: {strftime('%Y-%m-%d %H:%M:%S', localtime(time() + sleep_time))}"
        )
        await asyncio.sleep(sleep_time)
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
            logger.error(f"Error sending auto message: {e}")


REACTIONS = ["1️⃣", "2️⃣", "3️⃣"]


async def handle_reaction(message):
    """Handle reactions to messages"""

    message = await message.channel.fetch_message(message.id)  # Refresh message to get latest reactions

    reaction_counts = {}
    for reaction in message.reactions:
        if reaction.emoji in REACTIONS:
            reaction_counts[reaction.emoji] = reaction.count

    # Remove any emoji from REACTIONS if it already has 2 or more reactions
    available_reactions = [emoji for emoji in REACTIONS if reaction_counts.get(emoji, 0) < 2]
    if not available_reactions:
        logger.info("All REACTIONS have 2 or more reactions already.")
        return None

    try:
        await message.add_reaction(random.choice(available_reactions))
        return time()  # Return current timestamp on success
    except Exception as e:
        logger.error(f"Error adding reaction: {e}")
        return None  # Return None on error
