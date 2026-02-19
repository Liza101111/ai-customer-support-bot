import json

from app import db
from app.embeddings import embed


def main() -> None:
    rows = db.fetch_faq_entries(lang="en")

    update = 0
    for r in rows:
        text = f"{r['question']} {r.get('tags') or ''}".strip()
        vec = embed(text)
        db.update_faq_embedding(r["id"], json.dumps(vec))
        update += 1

    print(f"Embedded {update} FAQ entries.")


if __name__ == "__main__":
    main()
