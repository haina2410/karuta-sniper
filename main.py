import random
from time import time
import discord
from dotenv import load_dotenv
import os
import asyncio
import logging
from utils import auto_message_task, handle_reaction
from datetime import datetime, timezone
from typing import Optional

from ocr_utils import ocr_cards_from_url, ocr_prints_from_url  # type: ignore

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
        # Stats
        self.ocr_uses = 0
        self.last_ocr_duration: Optional[float] = None

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

        # Test commands (developer utilities)
        if message.content.startswith("ocrinfo"):
            await message.channel.send(
                f"OCR uses: {self.ocr_uses} | Last duration: {self.last_ocr_duration:.2f}s"
                if self.last_ocr_duration is not None
                else f"OCR uses: {self.ocr_uses} | Last duration: n/a"
            )
            return

        if message.content.startswith("ocr "):
            # Usage: ocr <image_url> <card_count>
            parts = message.content.split()
            if len(parts) < 3:
                await message.channel.send("Usage: ocr <url> <card_count>")
                return
            url = parts[1]
            try:
                count = int(parts[2])
            except ValueError:
                await message.channel.send("card_count must be integer")
                return
            start = time()
            try:
                # Run name/series OCR and print OCR concurrently
                cards_task = asyncio.create_task(ocr_cards_from_url(url, count))
                prints_task = asyncio.create_task(ocr_prints_from_url(url, count))
                cards, prints = await asyncio.gather(cards_task, prints_task, return_exceptions=True)

                self.ocr_uses += 1
                self.last_ocr_duration = time() - start

                # Handle potential exceptions individually
                if isinstance(cards, Exception):
                    raise cards
                if isinstance(prints, Exception):
                    logger.warning(f"Print OCR failed (continuing without prints): {prints}")
                    prints = []

                prints_by_index = {p['index']: p for p in prints if isinstance(p, dict)}

                if not cards:
                    await message.channel.send("No cards parsed (empty result)")
                    return

                lines = []
                for c in cards:
                    p = prints_by_index.get(c['index'])
                    if p and p.get('print_number') is not None:
                        lines.append(
                            f"#{c['index']+1}: {c['series']} - {c['name']} | print={p['print_number']} ed={p['edition']}"
                        )
                    else:
                        lines.append(
                            f"#{c['index']+1}: {c['series']} - {c['name']} | print=?"
                        )

                await message.channel.send(
                    f"Parsed {len(cards)} cards in {self.last_ocr_duration:.2f}s:\n" + "\n".join(lines)
                )
            except Exception as e:
                self.last_ocr_duration = time() - start
                await message.channel.send(f"OCR failed: {e}")
            return

        if message.content.startswith("ocrprint "):
            parts = message.content.split()
            if len(parts) < 3:
                await message.channel.send("Usage: ocrprint <url> <card_count>")
                return
            url = parts[1]
            try:
                count = int(parts[2])
            except ValueError:
                await message.channel.send("card_count must be integer")
                return
            start = time()
            try:
                pe = await ocr_prints_from_url(url, count)
                self.ocr_uses += 1
                self.last_ocr_duration = time() - start
                lines = [
                    f"#{c['index']+1}: print={c['print_number']} edition={c['edition']} raw='{c['raw']}'"
                    for c in pe
                ]
                await message.channel.send(
                    f"Parsed print data in {self.last_ocr_duration:.2f}s:\n" + "\n".join(lines)
                )
            except Exception as e:
                self.last_ocr_duration = time() - start
                await message.channel.send(f"OCR print failed: {e}")
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

        if "dropping" in message.content and "cards" in message.content:
            # Attempt to parse number of cards
            import re

            match = re.search(r"dropping (\d+) cards", message.content)
            card_count = int(match.group(1)) if match else 3

            # Defer OCR start: only schedule reaction sequence (which will OCR within its wait window)
            if card_count == 3:
                logger.info("Detected card drop message (3 cards), scheduling reaction sequence with deferred OCR")
                asyncio.create_task(self._reaction_sequence(message))

    async def _reaction_sequence(self, message: discord.Message):
        """Encapsulate the delayed reaction workflow with locking and rarity selection.

        Enhancement: attempt to pick the card with the lowest print number if
        an attachment is available and print OCR succeeds; otherwise default
        to first-available numeric reaction logic in `handle_reaction`.
        """
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

                # Kick off OCR concurrently during the waiting window to utilize idle time.
                ocr_task = None
                if message.attachments:
                    try:
                        attachment = message.attachments[0]
                        ocr_task = asyncio.create_task(ocr_prints_from_url(attachment.url, 3))
                    except Exception:
                        logger.exception("Failed to start OCR task for prints")

                # Wait before attempting number reaction (window for other players to react)
                await asyncio.sleep(random.uniform(38, 48))

                # Attempt rarity-based selection (lowest print number) using OCR result if available
                chosen_emoji = None
                if ocr_task:
                    try:
                        pe = await ocr_task
                        valid = [c for c in pe if c.get("print_number") is not None]
                        if valid:
                            target = min(valid, key=lambda c: c["print_number"])
                            idx = target["index"]
                            emoji_map = ["1️⃣", "2️⃣", "3️⃣"]
                            if 0 <= idx < len(emoji_map):
                                chosen_emoji = emoji_map[idx]
                                m = f"Rarity selection: chose index {idx+1} with print {target['print_number']} edition {target['edition']} (precomputed during wait)"
                                logger.info(m)
                                asyncio.create_task(message.channel.send(m))
                    except Exception:
                        logger.exception("Concurrent print OCR failed; falling back to default reaction heuristic")

                if chosen_emoji:
                    try:
                        await message.add_reaction(chosen_emoji)
                        reaction_time = time()
                    except Exception:
                        logger.exception("Failed adding chosen rarity emoji; fallback to default handler")
                        reaction_time = await handle_reaction(message)
                else:
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

    async def _ocr_and_log_cards(self, url: str, card_count: int):
        start = time()
        try:
            cards = await ocr_cards_from_url(url, card_count)
            self.ocr_uses += 1
            self.last_ocr_duration = time() - start
            if cards:
                preview = ", ".join(
                    f"{c['series']} - {c['name']}" for c in cards[: min(3, len(cards))]
                )
                logger.info(
                    f"OCR parsed {len(cards)} cards in {self.last_ocr_duration:.2f}s: {preview}{'...' if len(cards)>3 else ''}"
                )
            else:
                logger.info("OCR returned empty card list")
        except Exception:
            self.last_ocr_duration = time() - start
            logger.exception("OCR failed for drop attachment")


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
