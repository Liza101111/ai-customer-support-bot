import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DB_PATH = PROJECT_DIR / "support_bot.db"


FAQS = [
    {
        "question": "How do I request a refund?",
        "answer": "To request a refund, go to your Orders page, open the order, and click 'Request refund'. If you cannot find the option, contact support with your order number.",
        "tags": "refund,payment",
        "language": "en",
    },
    {
        "question": "Where is my order?",
        "answer": "You can track your order in Orders → Track shipment. If your tracking hasn’t updated in 48 hours, share your order number and we’ll check it.",
        "tags": "shipping,tracking,order",
        "language": "en",
    },
    {
        "question": "How long does delivery take?",
        "answer": "Delivery usually takes 3–7 business days depending on your location. You can see the estimated delivery date on your order confirmation.",
        "tags": "shipping,delivery",
        "language": "en",
    },
]


def main():

    with sqlite3.connect(DB_PATH, timeout=10) as conn:

        # Enforce foreign key constraints for this connection
        conn.execute("PRAGMA foreign_keys = ON;")

        # Skip duplicates: if the same question+language already exists, don't insert again.
        for f in FAQS:
            conn.execute(
                """
                INSERT OR IGNORE INTO faq_entries (question, answer, tags, language, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (f["question"], f["answer"], f["tags"], f["language"]),
            )

    print("Seeded FAQ entries successfully.")


if __name__ == "__main__":
    main()
