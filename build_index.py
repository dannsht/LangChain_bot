import os
import json
import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


class CliBot:
    def __init__(self, model_name: str, system_prompt: str):

        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data"
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)


        faq_path = data_dir / "faq.json"
        orders_path = data_dir / "orders.json"

        with faq_path.open("r", encoding="utf-8") as f:
            self.faq = json.load(f)

        with orders_path.open("r", encoding="utf-8") as f:
            self.orders = json.load(f)


        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=0,
            request_timeout=15,
        )


        self.store: dict[str, InMemoryChatMessageHistory] = {}


        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )


        self.chain = self.prompt | self.chat_model

        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )


        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = logs_dir / f"session_{timestamp}.jsonl"


        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }



    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def log_jsonl(self, record: dict):
        """Пишем одну строку JSON в session_*.jsonl."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



    def get_base_answer(self, user_text: str) -> str | None:
        """
        Простейший матчинг по FAQ:
        ищем вопрос из faq.json, который наибольшей длиной
        входит в текст пользователя (и наоборот).
        """
        text = user_text.lower()
        best = None
        best_len = 0

        for item in self.faq:
            q = item.get("q", "").lower()
            if not q:
                continue


            if q in text or text in q:
                if len(q) > best_len:
                    best = item
                    best_len = len(q)

        if best:
            return best.get("a")

        return None



    def get_order_status(self, order_id: str) -> str:
        """
        Читает статус заказа из orders.json (загружен в self.orders).
        Возвращает текст.
        """
        order = self.orders.get(order_id)
        if not order:
            return (
                f"Я не нашёл заказ с номером {order_id}. "
                f"Проверь, пожалуйста, номер или уточни его у поддержки."
            )

        status = order.get("status")

        if status == "in_transit":
            eta = order.get("eta_days")
            carrier = order.get("carrier", "служба доставки")
            if eta is not None:
                return (
                    f"Заказ {order_id} в пути с {carrier}. "
                    f"Ориентировочно будет доставлен в течение {eta} дн."
                )
            else:
                return f"Заказ {order_id} в пути. Ожидайте доставку в ближайшие дни."

        if status == "delivered":
            delivered_at = order.get("delivered_at")
            if delivered_at:
                return f"Заказ {order_id} уже доставлен {delivered_at}. Спасибо, что выбрали нас!"
            else:
                return f"Заказ {order_id} уже доставлен. Спасибо, что выбрали нас!"

        if status == "processing":
            note = order.get("note")
            if note:
                return f"Заказ {order_id} сейчас в обработке: {note}"
            else:
                return f"Заказ {order_id} сейчас в обработке на складе."

        # fallback, если что-то нестандартное
        return f"По заказу {order_id} вижу статус: {status}. Подробности пока недоступны."



    def __call__(self, session_id: str):
        print(
            "Чат-бот запущен! Можете задавать вопросы.\n"
            " - Для выхода введите 'выход'.\n"
            " - Для очистки контекста введите 'сброс'.\n"
            " - Для статуса заказа используйте: /order 12345\n"
        )


        self.log_jsonl({"event": "session_start", "session_id": session_id})

        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break

            if not user_text:
                continue


            self.log_jsonl({"role": "user", "content": user_text})

            msg_lower = user_text.lower()


            if msg_lower in ("выход", "стоп", "конец"):
                goodbye = "До свидания!"
                print(f"Бот: {goodbye}")
                self.log_jsonl({"role": "assistant", "content": goodbye})
                break

            if msg_lower == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                reply = "Контекст диалога очищен."
                print(f"Бот: {reply}")
                self.log_jsonl({"role": "assistant", "content": reply})
                continue

            if msg_lower in ("покажи историю", "история", "история диалога"):
                if session_id in self.store:
                    history = self.store[session_id].messages
                    print("История диалога:")
                    for message in history:
                        print(f"{message.type.capitalize()}: {message.content}")
                else:
                    print("История пуста.")

                continue


            if user_text.startswith("/order"):
                parts = user_text.split()
                if len(parts) < 2:
                    bot_reply = "Укажи номер заказа после команды, например: /order 12345."
                else:
                    order_id = parts[1]
                    bot_reply = self.get_order_status(order_id)

                print(f"Бот: {bot_reply}")
                self.log_jsonl({"role": "assistant", "content": bot_reply})
                continue


            faq_answer = self.get_base_answer(user_text)
            if faq_answer:
                bot_reply = faq_answer
                print(f"Бот: {bot_reply}")
                self.log_jsonl({"role": "assistant", "content": bot_reply})

                continue


            try:
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}},
                )
            except Exception as e:
                err_msg = f"[Ошибка] {e}"
                print(err_msg)
                self.log_jsonl(
                    {"role": "assistant", "content": err_msg, "error": True}
                )
                continue

            bot_reply = response.content.strip()
            print(f"Бот: {bot_reply}")


            usage_raw = getattr(response, "response_metadata", None)
            usage = None
            if isinstance(usage_raw, dict):
                token_usage = usage_raw.get("token_usage") or usage_raw.get("usage")
                if isinstance(token_usage, dict):

                    prompt_tokens = (
                        token_usage.get("prompt_tokens")
                        or token_usage.get("input_tokens")
                        or 0
                    )
                    completion_tokens = (
                        token_usage.get("completion_tokens")
                        or token_usage.get("output_tokens")
                        or 0
                    )
                    total_tokens = token_usage.get("total_tokens") or (
                        prompt_tokens + completion_tokens
                    )

                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }


                    self.total_usage["prompt_tokens"] += prompt_tokens
                    self.total_usage["completion_tokens"] += completion_tokens
                    self.total_usage["total_tokens"] += total_tokens


            record = {"role": "assistant", "content": bot_reply}
            if usage:
                record["usage"] = usage
            self.log_jsonl(record)


        self.log_jsonl(
            {"event": "session_end", "session_id": session_id, "usage": self.total_usage}
        )


if __name__ == "__main__":

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    brand = os.getenv("BRAND_NAME", "Shoply")


    system_prompt = f"""
Ты вежливый и краткий ассистент службы поддержки интернет-магазина {brand}.
Отвечай по делу, без лишней воды.

Если вопрос прямо покрывается FAQ, отвечай строго в рамках этих правил.
Если информации о политике магазина нет, честно говори, что не можешь ответить точно.
Никаких выдуманных скидок, условий или статусов заказов.
"""

    bot = CliBot(
        model_name=model,
        system_prompt=system_prompt.strip(),
    )
    bot("user_123")
