"""Create inspectable JSON chunks locally. No API calls or dependencies.

Run: python3 scripts/prepare_chunks.py
The character budget is provisional, NOT an embedding-model token limit.
Before embedding, validate sizes with the selected model's tokenizer.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def serialized_size(content):
    # Use this same compact JSON representation when sending to an embedder.
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))


def make_chunk(chunk_id, content, metadata):
    return {"chunk_id": chunk_id, "content": content, "metadata": metadata}


def restaurant_chunks(restaurant, max_chars):
    restaurant_id = restaurant["restaurant_id"]
    metadata = {
        "source_file": "data/restaurants.json",
        "restaurant_id": restaurant_id,
        "name": restaurant["name"],
        "city": restaurant["city"],
        "locality": restaurant["locality"],
        "cuisines": restaurant["cuisines"],
        "approx_cost_for_two_inr": restaurant["approx_cost_for_two_inr"],
        "rating_out_of_5": restaurant["rating_out_of_5"],
        "valet": restaurant["valet"],
        "demo_fields": ["rating_out_of_5", "opening_hours", "valet"],
    }
    if serialized_size(restaurant) <= max_chars:
        return [make_chunk(
            f"{restaurant_id}:profile", restaurant,
            {**metadata, "chunk_type": "restaurant"},
        )]

    # Keep the profile intact; split only the menu, never cut a dish name.
    profile = {key: value for key, value in restaurant.items() if key != "menu_items"}
    if serialized_size(profile) > max_chars:
        raise ValueError(f"Profile for {restaurant_id} exceeds the character budget.")
    chunks = [make_chunk(
        f"{restaurant_id}:profile", profile,
        {**metadata, "chunk_type": "restaurant", "menu_in_separate_chunks": True},
    )]
    identity = {
        "restaurant_id": restaurant_id,
        "name": restaurant["name"],
        "locality": restaurant["locality"],
    }
    menu_batch = []

    def append_menu(items):
        chunks.append(make_chunk(
            f"{restaurant_id}:menu:{len(chunks):03d}",
            {**identity, "menu_items": items},
            {**metadata, "chunk_type": "menu"},
        ))

    for dish in restaurant["menu_items"]:
        candidate = {**identity, "menu_items": menu_batch + [dish]}
        if serialized_size(candidate) > max_chars:
            if menu_batch:
                append_menu(menu_batch)
                menu_batch = []
            if serialized_size({**identity, "menu_items": [dish]}) > max_chars:
                raise ValueError(f"A single menu item for {restaurant_id} exceeds the budget.")
        menu_batch.append(dish)
    if menu_batch:
        append_menu(menu_batch)
    return chunks


def prepare_chunks(restaurants, faqs, max_chars=3000):
    chunks = []
    for restaurant in restaurants:
        chunks.extend(restaurant_chunks(restaurant, max_chars))
    for faq in faqs:
        if serialized_size(faq) > max_chars:
            raise ValueError(f"FAQ {faq['faq_id']} exceeds the character budget.")
        chunks.append(make_chunk(faq["faq_id"], faq, {
            "chunk_type": "faq", "source_file": "data/faqs.json",
            "faq_id": faq["faq_id"],
        }))
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs; check source record IDs.")
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=3000,
                        help="Maximum compact JSON content length (default: 3000 characters)")
    args = parser.parse_args()
    if args.max_chars < 1:
        parser.error("--max-chars must be positive")
    restaurants = json.loads((ROOT / "data/restaurants.json").read_text())
    faqs = json.loads((ROOT / "data/faqs.json").read_text())
    chunks = prepare_chunks(restaurants, faqs, args.max_chars)
    output = ROOT / "data/chunks.json"
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n")
    for kind in ("restaurant", "menu", "faq"):
        print(f"{kind}: {sum(c['metadata']['chunk_type'] == kind for c in chunks)} chunks")
    print(f"Saved {len(chunks)} chunks to {output}")


if __name__ == "__main__":
    main()
