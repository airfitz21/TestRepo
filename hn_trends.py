#!/usr/bin/env python3
"""
Fetch the top 5 trending topics from Hacker News and save them to trends.txt.

Uses the official Hacker News Firebase API (no external dependencies required).
"""

import json
import urllib.request
from datetime import datetime

HN_API = "https://hacker-news.firebaseio.com/v0"
OUTPUT_FILE = "trends.txt"
TOP_N = 5


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def fetch_top_stories(n: int) -> list[dict]:
    ids = fetch_json(f"{HN_API}/topstories.json")[:n]
    stories = []
    for story_id in ids:
        item = fetch_json(f"{HN_API}/item/{story_id}.json")
        stories.append({
            "title": item.get("title", "(no title)"),
            "url":   item.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
            "score": item.get("score", 0),
            "by":    item.get("by", "unknown"),
        })
    return stories


def save_trends(stories: list[dict], path: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Hacker News — Top {TOP_N} Trending Topics",
        f"Fetched: {timestamp}",
        "=" * 60,
    ]
    for i, s in enumerate(stories, 1):
        lines += [
            f"\n{i}. {s['title']}",
            f"   Score : {s['score']} points  |  By: {s['by']}",
            f"   URL   : {s['url']}",
        ]
    lines.append("")          # trailing newline

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print(f"Fetching top {TOP_N} stories from Hacker News...")
    stories = fetch_top_stories(TOP_N)
    save_trends(stories, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}\n")

    for i, s in enumerate(stories, 1):
        print(f"{i}. [{s['score']}pts] {s['title']}")


if __name__ == "__main__":
    main()
