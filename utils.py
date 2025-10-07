import asyncio
import random
import logging
from time import time, strftime, localtime
from typing import Optional
import discord

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


async def handle_reaction(message: discord.Message) -> Optional[float]:
    """Attempt to add one of the numeric reactions if its current count < 2.

    Returns timestamp (float) if a reaction was successfully added, else None.
    """
    try:
        # Refetch to mitigate stale reaction snapshot after long sleeps
        message = await message.channel.fetch_message(message.id)
    except Exception as e:
        logger.error(f"Failed to refetch message before reacting: {e}")
        return None

    reaction_counts: dict[str, int] = {}
    for reaction in message.reactions:
        if reaction.emoji in REACTIONS:
            reaction_counts[reaction.emoji] = reaction.count

    # Deterministic selection: first available
    chosen = next(
        (emoji for emoji in REACTIONS if reaction_counts.get(emoji, 0) < 2), None
    )
    if not chosen:
        logger.info("All target reactions already at limit (>=2).")
        return None

    try:
        await message.add_reaction(chosen)
        return time()
    except discord.Forbidden:
        logger.error("Forbidden: missing permission to add reactions.")
    except discord.NotFound:
        logger.error("Message not found when adding reaction.")
    except discord.HTTPException as e:
        logger.error(f"HTTP error adding reaction: {e}")
    except Exception as e:
        logger.error(f"Unexpected error adding reaction: {e}")
    return None
