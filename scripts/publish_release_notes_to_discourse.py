#!/usr/bin/env python3

import os

import requests


def load_config() -> dict[str, str]:
    env_config = {
        "DISCOURSE_URL": os.getenv("DISCOURSE_URL"),
        "DISCOURSE_API_KEY": os.getenv("DISCOURSE_API_KEY"),
        "DISCOURSE_USERNAME": os.getenv("DISCOURSE_USERNAME"),
        "DISCOURSE_CATEGORY": os.getenv("DISCOURSE_CATEGORY"),
        # Release information from GitHub
        "RELEASE_TAG": os.getenv("RELEASE_TAG"),
        "RELEASE_BODY": os.getenv("RELEASE_BODY"),
        "RELEASE_URL": os.getenv("RELEASE_URL"),
        "REPO_NAME": os.getenv("REPO_NAME"),
    }

    missing_env_values = {key: value for key, value in env_config.items() if value is None}
    if missing_env_values:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing_env_values.keys())}"
        )
    return env_config


def find_category_id(config: dict[str, str]) -> int:
    headers = {
        "Api-Key": config["DISCOURSE_API_KEY"],
        "Api-Username": config["DISCOURSE_USERNAME"],
        "Content-Type": "application/json",
    }

    category_to_find = config["DISCOURSE_CATEGORY"].lower()
    url = f"{config['DISCOURSE_URL']}/categories.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("Error fetching categories")
        raise

    if data.get("category_list") and data["category_list"].get("categories"):
        categories = data["category_list"]["categories"]

        for category in categories:
            cat_id = category.get("id")
            cat_name = category.get("name")
            if cat_name.lower() == category_to_find:
                return int(cat_id)

    raise ValueError(f"Category '{category_to_find}' not found")


def format_release_content(config: dict[str, str]) -> tuple[str, str]:
    title = f"🚀 Release {config['RELEASE_TAG']}"
    repo_name = config["REPO_NAME"].split("/")[1]
    content = f"""A new release of **{repo_name}** is now available!

## 📦 Release Information

- **Version:** `{config["RELEASE_TAG"]}`
- **Repository:** [{config["REPO_NAME"]}](https://github.com/{config["REPO_NAME"]})
- **Release Page:** {config["RELEASE_URL"]}
- Note: It may take some time for the release to appear on PyPI and conda-forge.

## 📋 Release Notes

{config["RELEASE_BODY"]}

---

*This post was automatically generated from the GitHub release.*
"""

    return title, content


def publish_release_to_discourse(config: dict[str, str]) -> bool:
    print("🎯 GitHub Release to Discourse Publisher")
    print(f"Release: {config['RELEASE_TAG']}")
    print(f"Repository: {config['REPO_NAME']}")
    print(f"Target Forum: {config['DISCOURSE_URL']}")
    print(f"Target Category: {config['DISCOURSE_CATEGORY']}")
    print("-" * 50)

    category_id = find_category_id(config)
    print(f"Publishing to category: {config['DISCOURSE_CATEGORY']} (ID: {category_id})")

    # Format the release content
    title, content = format_release_content(config)

    # Create the topic data
    topic_data = {"title": title, "raw": content, "category": category_id}

    # Post to Discourse
    headers = {
        "Api-Key": config["DISCOURSE_API_KEY"],
        "Api-Username": config["DISCOURSE_USERNAME"],
        "Content-Type": "application/json",
    }
    url = f"{config['DISCOURSE_URL']}/posts.json"

    try:
        response = requests.post(url, headers=headers, json=topic_data)
        response.raise_for_status()

        data = response.json()
        topic_id = data.get("topic_id")
        post_id = data.get("id")

        print("✅ Release published successfully!")
        print(f"Topic ID: {topic_id}")
        print(f"Post ID: {post_id}")
        print(f"URL: {config['DISCOURSE_URL']}/t/{topic_id}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error publishing release: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            try:
                error_data = e.response.json()
                print(f"Error details: {error_data}")
            except Exception:
                print(f"Response content: {e.response.text}")
        raise


if __name__ == "__main__":
    config = load_config()
    publish_release_to_discourse(config)
