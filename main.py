import random
from time import time
import discord
from dotenv import load_dotenv
import os
import asyncio
import logging
from utils import auto_message_task, handle_reaction
from datetime import datetime, timezone

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = 1119245745212624926  # Use env var with fallback
KARUTA_ID = 646937666251915264

# Reconnection configuration
MAX_RECONNECT_ATTEMPTS = 10
BASE_RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_DELAY = 300  # 5 minutes max

if not TOKEN:
    raise Exception("No token found in environment variables")


LATENCY_WARN_THRESHOLD = 2.0  # seconds difference between created_at and processing
EVENT_LOOP_LAG_WARN = 0.5  # seconds of scheduler delay warning


class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_react_time = 0
        self.reconnect_attempts = 0
        self.auto_task_started = False
        # Lock to prevent overlapping reaction workflows
        self._reaction_lock = asyncio.Lock()

    async def on_ready(self):
        logger.info(f"Logged on as {self.user}")
        self.reconnect_attempts = 0  # Reset on successful connection

        # Only start the auto message task once
        if not self.auto_task_started:
            asyncio.create_task(auto_message_task(self, CHANNEL_ID))
            self.auto_task_started = True
            logger.info("Auto message task started")
            # Start loop health monitor
            asyncio.create_task(self._loop_health_monitor())

    async def on_disconnect(self):
        logger.warning("Bot disconnected from Discord")

    async def on_resumed(self):
        logger.info("Bot connection resumed")
        self.reconnect_attempts = 0

    async def on_message(self, message):
        # only respond to channel
        if message.channel.id != CHANNEL_ID:
            return

        if message.content.startswith("na"):
            # remove na and replicate the rest
            content = message.content[2:].strip()

            # Fire-and-forget the send so we don't block the event dispatch path
            async def _echo():
                try:
                    await message.channel.send(content)
                    logger.info("Replicated message after 'na': " + content)
                except Exception:
                    logger.exception("Failed to replicate message")

            asyncio.create_task(_echo())
            return

        # Only respond to messages from Karuta bot
        if message.author.id != KARUTA_ID:
            return

        # Instrument processing latency (how far behind we processed this event)
        try:
            created_ts = message.created_at  # timezone-aware datetime (UTC)
            now_ts = datetime.now(timezone.utc)
            behind = (now_ts - created_ts).total_seconds()
            if behind > LATENCY_WARN_THRESHOLD:
                logger.warning(
                    f"on_message processing delay: {behind:.2f}s (websocket/backpressure or event loop lag)"
                )
        except Exception:
            logger.debug("Failed to compute message processing latency", exc_info=True)

        if "dropping 3 cards" in message.content:
            logger.info("Detected card drop message, creating reaction task")
            # Launch the reaction workflow as a task so on_message returns quickly
            asyncio.create_task(self._reaction_sequence(message))

    async def _reaction_sequence(self, message: discord.Message):
        """Encapsulate the delayed reaction workflow with locking and error handling."""
        async with self._reaction_lock:
            # Cooldown check (10 minutes)
            current_time = time()
            if (
                self.last_react_time > 0
                and current_time - self.last_react_time < 60 * 10
            ):
                logger.info("Cooldown active, skipping reaction")
                return

            try:
                # Short random delay before initial clock reaction
                await asyncio.sleep(random.uniform(2, 3))

                # Add clock reaction in a monitored task
                clock_task = asyncio.create_task(message.add_reaction("🕒"))
                clock_task.add_done_callback(
                    lambda fut: logger.error(
                        f"Clock reaction failed: {fut.exception()}"
                    )
                    if fut.exception()
                    else None
                )

                # Wait before attempting number reaction
                await asyncio.sleep(random.uniform(38, 48))

                reaction_time = await handle_reaction(message)
                if reaction_time:
                    self.last_react_time = reaction_time
                    logger.info("Successfully reacted to card drop")

                    # Short delay before sending follow-up command
                    await asyncio.sleep(random.uniform(4, 8))
                    channel = self.get_channel(CHANNEL_ID)
                    if channel:
                        await channel.send("kt a")
                        logger.info("Sent 'kt a' command")
                    else:
                        logger.error(f"Channel with ID {CHANNEL_ID} not found")
            except Exception:
                logger.exception("Error during reaction workflow")

    async def _loop_health_monitor(self):
        """Periodically check for event loop scheduling delays (indicative of blocking code)."""
        loop = asyncio.get_running_loop()
        # Expectation reference point
        expected = loop.time() + 1.0
        while not self.is_closed():
            await asyncio.sleep(1.0)
            now = loop.time()
            lag = now - expected
            if lag > EVENT_LOOP_LAG_WARN:
                logger.warning(
                    f"Event loop lag detected: {lag:.3f}s (something blocked the loop)"
                )
            expected = now + 1.0


async def run_with_reconnect():
    """Run bot with automatic reconnection logic"""
    client = MyClient()
    reconnect_attempts = 0

    while True:
        try:
            logger.info("Starting bot...")
            await client.start(TOKEN)
        except discord.ConnectionClosed:
            logger.warning("Connection closed by Discord")
        except discord.LoginFailure:
            logger.error("Invalid token - cannot reconnect")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        # Check if we should attempt reconnection
        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.error(
                f"Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Giving up."
            )
            break

        reconnect_attempts += 1

        # Calculate exponential backoff delay
        delay = min(
            BASE_RECONNECT_DELAY * (2 ** (reconnect_attempts - 1)), MAX_RECONNECT_DELAY
        )

        logger.info(
            f"Reconnection attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} in {delay} seconds..."
        )
        await asyncio.sleep(delay)

        # Reset the client for reconnection
        if not client.is_closed():
            await client.close()
        client = MyClient()


if __name__ == "__main__":
    try:
        asyncio.run(run_with_reconnect())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
