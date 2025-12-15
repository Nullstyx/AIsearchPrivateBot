import os
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Set
from pathlib import Path

from dotenv import load_dotenv
import telebot
from telebot import TeleBot, types as tb_types
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config_path = Path('config.env')
if not config_path.exists():
    logger.error(f"Config file {config_path} not found!")
else:
    logger.info(f"Loading config from {config_path.absolute()}")

load_dotenv('config.env', override=True)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'

DATA_DIR = Path('user_data')
DATA_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_MESSAGES = 100
MAX_CONTEXT_TOKENS = 12000
TOKENS_PER_CHAR = 0.25

DEFAULT_SYSTEM_PROMPT = ""

ALLOWED_CHAT_IDS: Set[int] = set()
ALLOWED_USER_MODES: Dict[int, Set[str]] = {}


def _load_user_permissions():
    global ALLOWED_CHAT_IDS, ALLOWED_USER_MODES

    if not config_path.exists():
        logger.warning("Config file not found, no user restrictions will be applied.")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue

                if not line.startswith('User'):
                    continue

                if '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if not key.startswith('User') or not value:
                    continue

                parts = value.split('=', 1)
                chat_id_str = parts[0].strip()
                modes_str = parts[1].strip() if len(parts) > 1 else ""

                try:
                    chat_id = int(chat_id_str)
                except ValueError:
                    logger.warning(f"Invalid chat_id in {key}: {chat_id_str}")
                    continue

                ALLOWED_CHAT_IDS.add(chat_id)

                allowed_modes: Set[str] = set()
                for ch in modes_str:
                    if ch in ('1', '2', '3'):
                        allowed_modes.add(ch)

                # even если строка режимов пуста, сохраняем пустой сет, чтобы явно запретить все режимы
                ALLOWED_USER_MODES[chat_id] = allowed_modes

        logger.info(f"User access list loaded. Chats: {ALLOWED_CHAT_IDS}, raw mode masks: {ALLOWED_USER_MODES}")
    except Exception as e:
        logger.error(f"Error reading user permissions from config.env: {e}")


class BotMode(Enum):
    STANDARD = "standard"
    PRO = "pro"
    REASONING = "reasoning"


_load_user_permissions()


class UserState:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.mode = BotMode.STANDARD
        self.message_history: List[Dict[str, str]] = []
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.waiting_for_prompt = False
        self.load_from_disk()
    
    def get_data_file_path(self) -> Path:
        return DATA_DIR / f'user_{self.user_id}.json'
    
    def save_to_disk(self):
        try:
            data = {
                'mode': self.mode.value,
                'message_history': self.message_history,
                'system_prompt': self.system_prompt
            }
            with open(self.get_data_file_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving user data for user {self.user_id}: {e}")
    
    def load_from_disk(self):
        try:
            file_path = self.get_data_file_path()
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'mode' in data:
                        try:
                            self.mode = BotMode(data['mode'])
                        except ValueError:
                            self.mode = BotMode.STANDARD
                    if 'message_history' in data:
                        self.message_history = data['message_history']
                    if 'system_prompt' in data:
                        self.system_prompt = data['system_prompt']
        except Exception as e:
            logger.error(f"Error loading user data for user {self.user_id}: {e}")
    
    def get_messages_with_system_prompt(self) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt and self.system_prompt.strip():
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        messages.extend(self.message_history)
        return messages

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) * TOKENS_PER_CHAR)

    def trim_context_if_needed(self):
        if not self.message_history:
            return
        
        total_tokens = sum(
            self.estimate_tokens(msg.get("content", "")) 
            for msg in self.message_history
        )
        
        if total_tokens > MAX_CONTEXT_TOKENS:
            trimmed_history = []
            current_tokens = 0
            
            for msg in reversed(self.message_history):
                msg_tokens = self.estimate_tokens(msg.get("content", ""))
                if current_tokens + msg_tokens <= MAX_CONTEXT_TOKENS:
                    trimmed_history.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    break
            
            if len(trimmed_history) >= 2:
                self.message_history = trimmed_history
                logger.info(f"Context trimmed for user {self.user_id}: {len(self.message_history)} messages, ~{current_tokens} tokens")
        
        if len(self.message_history) > MAX_CONTEXT_MESSAGES:
            self.message_history = self.message_history[-MAX_CONTEXT_MESSAGES:]
            logger.info(f"Context trimmed by message count for user {self.user_id}: {len(self.message_history)} messages")


class PerplexityTelegramBot:
    def __init__(self):
        self.bot = TeleBot(TELEGRAM_BOT_TOKEN)
        self.user_states: Dict[int, UserState] = {}
        self.setup_handlers()

    def get_allowed_modes_for_chat(self, chat_id: int) -> List[BotMode]:
        if not ALLOWED_CHAT_IDS:
            return [BotMode.STANDARD, BotMode.PRO, BotMode.REASONING]

        mask = ALLOWED_USER_MODES.get(chat_id)
        if mask is None:
            return [BotMode.STANDARD, BotMode.PRO, BotMode.REASONING]

        if not mask:
            return []

        modes: List[BotMode] = []
        if '1' in mask:
            modes.append(BotMode.STANDARD)
        if '2' in mask:
            modes.append(BotMode.PRO)
        if '3' in mask:
            modes.append(BotMode.REASONING)
        return modes

    def is_allowed_user(self, chat_id: int) -> bool:
        if not ALLOWED_CHAT_IDS:
            return True
        is_allowed = chat_id in ALLOWED_CHAT_IDS
        if not is_allowed:
            logger.warning(f"Access denied for chat_id {chat_id}. Allowed: {ALLOWED_CHAT_IDS}")
        return is_allowed

    def get_user_state(self, user_id: int) -> UserState:
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState(user_id)
        return self.user_states[user_id]
    
    def save_user_state(self, user_id: int):
        if user_id in self.user_states:
            self.user_states[user_id].save_to_disk()

    def get_model_name(self, mode: BotMode) -> str:
        if mode == BotMode.STANDARD:
            return "sonar"
        elif mode == BotMode.PRO:
            return "sonar-pro"
        elif mode == BotMode.REASONING:
            return "sonar-reasoning"
        return "sonar"

    def setup_handlers(self):
        @self.bot.message_handler(func=lambda message: not self.is_allowed_user(message.chat.id))
        def handle_unauthorized(message):
            self.bot.reply_to(message, "❌ Доступ запрещен. Ваш chat_id не в списке разрешенных пользователей.")

        @self.bot.message_handler(func=lambda message: getattr(message.chat, "type", None) != "private")
        def handle_non_private(message):
            return

        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            user_state = self.get_user_state(message.from_user.id)
            user_state.load_from_disk()
            
            help_text = (
                "👋 Привет! Я бот с интеграцией Perplexity AI.\n\n"
                "📋 Основные команды:\n"
                "/mode - сменить режим работы\n"
                "/reset - сбросить контекст диалога\n"
                "/prompt - управление системным промптом\n\n"
                "🔧 Режимы работы:\n"
                "• Sonar - быстрый режим для обычных вопросов\n"
                "• Sonar Pro - более точный и детальный режим\n"
                "• Sonar-reasoning - режим с подробными рассуждениями\n\n"
                "📝 Контекст диалога:\n"
                "• Полное сохранение истории (до 100 сообщений или ~12000 токенов)\n"
                "• Автоматическая умная обрезка при превышении лимитов\n"
                "• Сохранение контекста при перезапуске бота\n\n"
                "💡 Просто отправьте мне вопрос, и я найду ответ!"
            )
            
            markup = tb_types.InlineKeyboardMarkup()
            markup.row(
                tb_types.InlineKeyboardButton("/mode", callback_data='cmd_mode'),
                tb_types.InlineKeyboardButton("/reset", callback_data='cmd_reset'),
                tb_types.InlineKeyboardButton("/prompt", callback_data='cmd_prompt')
            )
            markup.add(tb_types.InlineKeyboardButton(
                "⤴️ Страница разработчика",
                url="https://github.com/MrachniyTipchek"
            ))
            
            self.bot.reply_to(message, help_text, reply_markup=markup)

        @self.bot.message_handler(commands=['mode'])
        def handle_mode(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return

            user_state = self.get_user_state(message.from_user.id)
            allowed_modes = self.get_allowed_modes_for_chat(message.chat.id)
            if not allowed_modes:
                self.bot.reply_to(message, "Для вас не доступен ни один режим работы бота.")
                return

            markup = tb_types.InlineKeyboardMarkup()
            
            current_mode = user_state.mode
            check_standard = "✅ " if current_mode == BotMode.STANDARD else ""
            check_pro = "✅ " if current_mode == BotMode.PRO else ""
            check_reasoning = "✅ " if current_mode == BotMode.REASONING else ""

            if BotMode.STANDARD in allowed_modes:
                markup.add(tb_types.InlineKeyboardButton(f"{check_standard}Sonar", callback_data='mode_standard'))
            if BotMode.PRO in allowed_modes:
                markup.add(tb_types.InlineKeyboardButton(f"{check_pro}Sonar Pro", callback_data='mode_pro'))
            if BotMode.REASONING in allowed_modes:
                markup.add(tb_types.InlineKeyboardButton(f"{check_reasoning}Sonar-reasoning", callback_data='mode_reasoning'))
            
            mode_names = {
                BotMode.STANDARD: "Sonar",
                BotMode.PRO: "Sonar Pro",
                BotMode.REASONING: "Sonar-reasoning"
            }
            
            self.bot.send_message(
                message.chat.id,
                f"Текущий режим: {mode_names[user_state.mode]}\nВыберите новый режим:",
                reply_markup=markup
            )

        @self.bot.message_handler(commands=['reset'])
        def handle_reset(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            user_state = self.get_user_state(message.from_user.id)
            user_state.message_history = []
            user_state.save_to_disk()
            self.bot.reply_to(message, "Контекст диалога сброшен.")
        
        @self.bot.message_handler(commands=['prompt'])
        def handle_prompt(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            user_state = self.get_user_state(message.from_user.id)
            
            markup = tb_types.InlineKeyboardMarkup()
            markup.row(
                tb_types.InlineKeyboardButton("🔄 Сброс", callback_data='prompt_reset'),
                tb_types.InlineKeyboardButton("✏️ Изменить", callback_data='prompt_edit')
            )
            
            if user_state.system_prompt and user_state.system_prompt.strip():
                prompt_text = user_state.system_prompt
            else:
                prompt_text = "Системный промпт еще не указан!"
            
            self.bot.send_message(
                message.chat.id,
                f"📝 Системный промпт:\n\n{prompt_text}",
                reply_markup=markup
            )

        @self.bot.callback_query_handler(func=lambda call: not self.is_allowed_user(call.message.chat.id))
        def handle_unauthorized_callback(call):
            self.bot.answer_callback_query(call.id, "❌ Доступ запрещен", show_alert=True)

        @self.bot.callback_query_handler(func=lambda call: call.data.startswith('cmd_'))
        def handle_cmd_callback(call):
            if getattr(call.message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(call.message.chat.id):
                return

            if call.data == 'cmd_mode':
                user_state = self.get_user_state(call.from_user.id)
                markup = tb_types.InlineKeyboardMarkup()
                allowed_modes = self.get_allowed_modes_for_chat(call.message.chat.id)
                if not allowed_modes:
                    self.bot.answer_callback_query(call.id, "Для вас не доступен ни один режим.", show_alert=True)
                    return

                current_mode = user_state.mode
                check_standard = "✅ " if current_mode == BotMode.STANDARD else ""
                check_pro = "✅ " if current_mode == BotMode.PRO else ""
                check_reasoning = "✅ " if current_mode == BotMode.REASONING else ""

                if BotMode.STANDARD in allowed_modes:
                    markup.add(tb_types.InlineKeyboardButton(f"{check_standard}Sonar", callback_data='mode_standard'))
                if BotMode.PRO in allowed_modes:
                    markup.add(tb_types.InlineKeyboardButton(f"{check_pro}Sonar Pro", callback_data='mode_pro'))
                if BotMode.REASONING in allowed_modes:
                    markup.add(tb_types.InlineKeyboardButton(f"{check_reasoning}Sonar-reasoning", callback_data='mode_reasoning'))
                mode_names = {
                    BotMode.STANDARD: "Sonar",
                    BotMode.PRO: "Sonar Pro",
                    BotMode.REASONING: "Sonar-reasoning"
                }
                self.bot.answer_callback_query(call.id)
                self.bot.send_message(
                    call.message.chat.id,
                    f"Текущий режим: {mode_names[user_state.mode]}\nВыберите новый режим:",
                    reply_markup=markup
                )
            elif call.data == 'cmd_reset':
                user_state = self.get_user_state(call.from_user.id)
                user_state.message_history = []
                user_state.save_to_disk()
                self.bot.answer_callback_query(call.id, "Контекст диалога сброшен")
                self.bot.send_message(call.message.chat.id, "Контекст диалога сброшен.")
            elif call.data == 'cmd_prompt':
                user_state = self.get_user_state(call.from_user.id)
                markup = tb_types.InlineKeyboardMarkup()
                markup.row(
                    tb_types.InlineKeyboardButton("🔄 Сброс", callback_data='prompt_reset'),
                    tb_types.InlineKeyboardButton("✏️ Изменить", callback_data='prompt_edit')
                )
                if user_state.system_prompt and user_state.system_prompt.strip():
                    prompt_text = user_state.system_prompt
                else:
                    prompt_text = "Системный промпт еще не указан!"
                self.bot.answer_callback_query(call.id)
                self.bot.send_message(
                    call.message.chat.id,
                    f"📝 Системный промпт:\n\n{prompt_text}",
                    reply_markup=markup
                )

        @self.bot.callback_query_handler(func=lambda call: call.data.startswith('mode_'))
        def handle_mode_callback(call):
            if getattr(call.message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(call.message.chat.id):
                return

            user_state = self.get_user_state(call.from_user.id)
            allowed_modes = self.get_allowed_modes_for_chat(call.message.chat.id)
            
            if call.data == 'mode_standard':
                user_state.mode = BotMode.STANDARD
                mode_name = "Sonar"
            elif call.data == 'mode_pro':
                user_state.mode = BotMode.PRO
                mode_name = "Sonar Pro"
            elif call.data == 'mode_reasoning':
                user_state.mode = BotMode.REASONING
                mode_name = "Sonar-reasoning"
            else:
                return

            if allowed_modes and user_state.mode not in allowed_modes:
                self.bot.answer_callback_query(call.id, "Этот режим вам недоступен.", show_alert=True)
                return
            
            user_state.save_to_disk()
            
            markup = tb_types.InlineKeyboardMarkup()
            check_standard = "✅ " if user_state.mode == BotMode.STANDARD else ""
            check_pro = "✅ " if user_state.mode == BotMode.PRO else ""
            check_reasoning = "✅ " if user_state.mode == BotMode.REASONING else ""
            
            markup.add(tb_types.InlineKeyboardButton(f"{check_standard}Sonar", callback_data='mode_standard'))
            markup.add(tb_types.InlineKeyboardButton(f"{check_pro}Sonar Pro", callback_data='mode_pro'))
            markup.add(tb_types.InlineKeyboardButton(f"{check_reasoning}Sonar-reasoning", callback_data='mode_reasoning'))
            
            self.bot.answer_callback_query(call.id, f"Режим изменен на: {mode_name}")
            self.bot.edit_message_text(
                f"Текущий режим: {mode_name}\nВыберите новый режим:",
                call.message.chat.id,
                call.message.message_id,
                reply_markup=markup
            )
        
        @self.bot.callback_query_handler(func=lambda call: call.data == 'prompt_reset')
        def handle_prompt_reset(call):
            if not self.is_allowed_user(call.message.chat.id):
                return
            
            user_state = self.get_user_state(call.from_user.id)
            user_state.system_prompt = ""
            user_state.save_to_disk()
            
            markup = tb_types.InlineKeyboardMarkup()
            markup.row(
                tb_types.InlineKeyboardButton("🔄 Сброс", callback_data='prompt_reset'),
                tb_types.InlineKeyboardButton("✏️ Изменить", callback_data='prompt_edit')
            )
            
            prompt_text = "Системный промпт еще не указан!"
            
            self.bot.answer_callback_query(call.id, "Промпт сброшен")
            self.bot.edit_message_text(
                f"📝 Системный промпт:\n\n{prompt_text}",
                call.message.chat.id,
                call.message.message_id,
                reply_markup=markup
            )
        
        @self.bot.callback_query_handler(func=lambda call: call.data == 'prompt_edit')
        def handle_prompt_edit(call):
            if not self.is_allowed_user(call.message.chat.id):
                return
            
            user_state = self.get_user_state(call.from_user.id)
            user_state.waiting_for_prompt = True
            
            self.bot.answer_callback_query(call.id)
            self.bot.send_message(
                call.message.chat.id,
                "✏️ Отправьте новый системный промпт. Это инструкция, которая определяет роль и поведение бота.\n\n"
                "Пример: 'Ты опытный программист. Отвечай кратко и по делу.'\n\n"
                "Для отмены отправьте /cancel"
            )

        @self.bot.message_handler(commands=['cancel'])
        def handle_cancel(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            user_state = self.get_user_state(message.from_user.id)
            if hasattr(user_state, 'waiting_for_prompt') and user_state.waiting_for_prompt:
                user_state.waiting_for_prompt = False
                self.bot.reply_to(message, "Отмена изменения промпта.")
            else:
                self.bot.reply_to(message, "Нет активных операций для отмены.")

        @self.bot.message_handler(content_types=['text'])
        def handle_text(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            user_state = self.get_user_state(message.from_user.id)
            
            if hasattr(user_state, 'waiting_for_prompt') and user_state.waiting_for_prompt:
                user_state.system_prompt = message.text
                user_state.waiting_for_prompt = False
                user_state.save_to_disk()
                self.bot.reply_to(message, f"✅ Системный промпт обновлен:\n\n{message.text}")
                return
            
            self.process_message(message, message.text)

        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            self.bot.reply_to(
                message, 
                "К сожалению, Perplexity API не поддерживает обработку изображений напрямую. "
                "Отправьте текстовый вопрос или опишите изображение текстом."
            )

        @self.bot.message_handler(content_types=['document'])
        def handle_document(message):
            if getattr(message.chat, "type", None) != "private":
                return
            if not self.is_allowed_user(message.chat.id):
                return
            self.bot.reply_to(
                message,
                "К сожалению, Perplexity API не поддерживает обработку документов напрямую. "
                "Отправьте текстовый вопрос или опишите содержимое документа."
            )

    def process_message(self, message, user_text: str):
        user_state = self.get_user_state(message.from_user.id)
        
        self.bot.send_chat_action(message.chat.id, 'typing')
        
        try:
            user_state.message_history.append({
                "role": "user",
                "content": user_text
            })
            
            model = self.get_model_name(user_state.mode)
            
            messages = user_state.get_messages_with_system_prompt()
            
            json_data = {
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            if user_state.mode == BotMode.REASONING:
                json_data["temperature"] = 0.8
            
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    PERPLEXITY_API_URL,
                    headers=headers,
                    json=json_data
                )
                response.raise_for_status()
                data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                assistant_message = data["choices"][0]["message"]["content"]
                
                user_state.message_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                user_state.trim_context_if_needed()
                
                user_state.save_to_disk()
                
                if len(assistant_message) > 4000:
                    parts = [assistant_message[i:i+4000] for i in range(0, len(assistant_message), 4000)]
                    for i, part in enumerate(parts):
                        if i == 0:
                            self.bot.reply_to(message, part)
                        else:
                            self.bot.send_message(message.chat.id, part)
                else:
                    self.bot.reply_to(message, assistant_message)
            else:
                self.bot.reply_to(message, "Не удалось получить ответ от API.")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            self.bot.reply_to(message, f"Ошибка API: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            self.bot.reply_to(message, "Ошибка подключения к API. Попробуйте позже.")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.bot.reply_to(message, f"Ошибка обработки запроса: {str(e)}")

    def run(self):
        logger.info("Starting Perplexity bot...")
        if ALLOWED_CHAT_IDS:
            logger.info(f"Access restricted to {len(ALLOWED_CHAT_IDS)} users: {ALLOWED_CHAT_IDS}")
        else:
            logger.info("No chat_id specified - access open for all users")
        self.bot.infinity_polling()


if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN or not PERPLEXITY_API_KEY:
        print("ОШИБКА: Заполните TELEGRAM_BOT_TOKEN и PERPLEXITY_API_KEY в config.env!")
        print("1. Получите TELEGRAM_BOT_TOKEN у @BotFather")
        print("2. Получите PERPLEXITY_API_KEY на https://www.perplexity.ai/account/api/keys")
        print("3. Заполните значения в файле config.env")
    else:
        bot = PerplexityTelegramBot()
        bot.run()
