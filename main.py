import os
from notion_client import Client

notion = Client(auth=os.environ.get("NOTION_API_KEY"))

ROOT_PAGE_ID = "2f9a97f0fe3080ea8e2ff6a924e98ad9"

def get_block_children(block_id):
    children = []
    cursor = None

    while True:
        response = notion.blocks.children.list(
            block_id=block_id,
            start_cursor=cursor
        )
        children.extend(response["results"])

        if not response["has_more"]:
            break
        cursor = response["next_cursor"]

    return children

def extract_text_from_block(block):
    block_type = block["type"]

    if block_type not in block:
        return ""

    content = block[block_type]

    if "rich_text" in content:
        return "".join([t["plain_text"] for t in content["rich_text"]])

    if "title" in content:
        return content["title"]

    return ""

def get_page_title(page_id):
    try:
        page = notion.pages.retrieve(page_id)
        props = page.get("properties", {})

        for prop in props.values():
            if prop["type"] == "title":
                title_content = prop.get("title", [])
                if title_content:
                    return "".join([t["plain_text"] for t in title_content])
        return "Untitled"
    except:
        return "Untitled"

def process_page(page_id, path=[]):
    chunks = []
    title = get_page_title(page_id)
    current_path = path + [title]

    print(f"Processing: {' > '.join(current_path)}")

    blocks = get_block_children(page_id)

    current_chunk = {
        "text": "",
        "source_id": page_id,
        "path": current_path,
        "title": title
    }

    for block in blocks:
        block_type = block["type"]

        if block_type == "child_page":
            if current_chunk["text"].strip():
                chunks.append(current_chunk.copy())
                current_chunk["text"] = ""

            child_chunks = process_page(block["id"], current_path)
            chunks.extend(child_chunks)
            continue

        if block_type == "child_database":
            continue

        text = extract_text_from_block(block)

        if block_type in ["heading_1", "heading_2", "heading_3"]:
            if current_chunk["text"].strip():
                chunks.append(current_chunk.copy())
            current_chunk["text"] = f"## {text}\n"
        else:
            if text:
                current_chunk["text"] += text + "\n"

        if block.get("has_children") and block_type not in ["child_page", "child_database"]:
            toggle_children = get_block_children(block["id"])
            for child in toggle_children:
                child_text = extract_text_from_block(child)
                if child_text:
                    current_chunk["text"] += "  " + child_text + "\n"

    if current_chunk["text"].strip():
        chunks.append(current_chunk)

    return chunks

if __name__ == "__main__":
    print("Starting Notion ingestion...\n")

    all_chunks = process_page(ROOT_PAGE_ID)

    print(f"\n--- Done! Found {len(all_chunks)} chunks ---\n")

    for i, chunk in enumerate(all_chunks[:5]):
        print(f"[{i+1}] {' > '.join(chunk['path'])}")
        print(f"    {chunk['text'][:100]}...")
        print()

# Save chunks to file
import json
with open('chunks.json', 'w') as f:
    json.dump(all_chunks, f, indent=2)
print("Saved chunks to chunks.json")