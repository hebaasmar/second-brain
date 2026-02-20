"""
Notion Story Bank Sync
Pulls story notes from the Story Bank notebook in Notion,
chunks them by beat, and creates embeddings for semantic search.
"""

import os
import re
import json
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

# Story Bank notebook page ID
STORY_BANK_ID = "306a97f0-fe30-80e4-bc9c-f2d45af3e479"


def get_notion_client():
    token = os.getenv("NOTION_TOKEN")
    if not token:
        raise ValueError("NOTION_TOKEN not found in .env file")
    return Client(auth=token)


def get_story_notes(notion):
    """Find all notes linked to the Story Bank notebook."""
    # First get the page to find the property ID for "Add a note"
    page = notion.pages.retrieve(page_id=STORY_BANK_ID)
    props = page.get("properties", {})

    # Find the "Add a note" relation property
    relation_prop = props.get("Add a note", {})
    prop_id = relation_prop.get("id")

    if not prop_id:
        print("Could not find 'Add a note' property")
        print(f"Available properties: {list(props.keys())}")
        return []

    print(f"Found 'Add a note' property (id: {prop_id})")

    # Use property endpoint to get ALL related page IDs (handles pagination)
    note_ids = []
    cursor = None

    while True:
        kwargs = {"page_id": STORY_BANK_ID, "property_id": prop_id}
        if cursor:
            kwargs["start_cursor"] = cursor

        result = notion.pages.properties.retrieve(**kwargs)

        for item in result.get("results", []):
            if item.get("type") == "relation":
                note_ids.append(item["relation"]["id"])

        if not result.get("has_more"):
            break
        cursor = result.get("next_cursor")

    print(f"Found {len(note_ids)} story notes in Story Bank")
    return note_ids


def get_page_content(notion, page_id):
    """Get all text content from a Notion page."""
    blocks = []
    cursor = None

    while True:
        if cursor:
            response = notion.blocks.children.list(block_id=page_id, start_cursor=cursor)
        else:
            response = notion.blocks.children.list(block_id=page_id)

        blocks.extend(response["results"])

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    lines = []
    for block in blocks:
        block_type = block["type"]

        if block_type in ("paragraph", "bulleted_list_item", "numbered_list_item", "quote", "callout"):
            text = extract_rich_text(block[block_type].get("rich_text", []))
            if text:
                lines.append(text)

        elif block_type in ("heading_1", "heading_2", "heading_3"):
            text = extract_rich_text(block[block_type].get("rich_text", []))
            prefix = "#" * int(block_type[-1])
            if text:
                lines.append(f"{prefix} {text}")

        elif block_type == "toggle":
            text = extract_rich_text(block["toggle"].get("rich_text", []))
            if text:
                lines.append(text)
            if block.get("has_children"):
                child_text = get_page_content(notion, block["id"])
                if child_text:
                    lines.append(child_text)

    return "\n".join(lines)


def extract_rich_text(rich_text_array):
    """Extract plain text from Notion rich text array."""
    return "".join([rt.get("plain_text", "") for rt in rich_text_array])


def parse_story_into_beats(title, content):
    """Split a story note into individual beats for chunking."""
    chunks = []

    # Parse company and story name from title
    company = "Unknown"
    story_name = title
    if ":" in title:
        parts = title.split(":", 1)
        company = parts[0].strip()
        story_name = parts[1].strip()

    # Extract tags
    tags = []
    tag_match = re.search(r'Tags?:\s*(.+)', content, re.IGNORECASE)
    if tag_match:
        tags = [t.strip() for t in tag_match.group(1).split(",")]

    # Split on ## Beat headers
    beat_pattern = r'(?=## Beat \d+)'
    beat_sections = re.split(beat_pattern, content)

    beats = [s.strip() for s in beat_sections if s.strip() and re.match(r'## Beat \d+', s.strip())]

    if beats:
        for beat_text in beats:
            first_line = beat_text.split("\n")[0]
            beat_title = first_line.replace("##", "").strip()

            chunk = {
                "company": company,
                "story": story_name,
                "beat": beat_title,
                "tags": tags,
                "text": f"{company} | {story_name} | {beat_title}\n\n{beat_text}",
                "path": [company, story_name, beat_title]
            }
            chunks.append(chunk)
    else:
        # No structured beats yet (placeholder) - chunk whole thing
        if len(content.strip()) > 50:
            chunk = {
                "company": company,
                "story": story_name,
                "beat": "Full story",
                "tags": tags,
                "text": f"{company} | {story_name}\n\n{content}",
                "path": [company, story_name]
            }
            chunks.append(chunk)

    return chunks


def sync_story_bank():
    """Pull Story Bank from Notion, parse into beats, return chunks."""
    print("Connecting to Notion...")
    notion = get_notion_client()

    print("Fetching Story Bank notes...")
    note_ids = get_story_notes(notion)

    if not note_ids:
        print("No notes found in Story Bank.")
        return []

    all_chunks = []

    for note_id in note_ids:
        page = notion.pages.retrieve(page_id=note_id)
        title_prop = page.get("properties", {}).get("Name", {})

        if title_prop.get("type") == "title":
            title = extract_rich_text(title_prop.get("title", []))
        else:
            title = "Untitled"

        print(f"  Processing: {title}")

        content = get_page_content(notion, note_id)
        beats = parse_story_into_beats(title, content)
        all_chunks.extend(beats)
        print(f"    -> {len(beats)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(note_ids)} stories")
    return all_chunks


if __name__ == "__main__":
    chunks = sync_story_bank()

    if chunks:
        with open("story_bank_chunks.json", "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"\nSaved {len(chunks)} chunks to story_bank_chunks.json")

        print("\n--- Preview ---")
        for c in chunks[:3]:
            print(f"\n[{c['company']} | {c['story']}]")
            print(f"Beat: {c.get('beat', 'N/A')}")
            print(f"Text: {c['text'][:200]}...")
