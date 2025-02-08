import os
import requests
import time
import json

# Wikipedia API endpoint to fetch page images
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Add this constant near the top with other constants
HEADERS = {
    'User-Agent': 'Python/3 - BillionaireImageCollector/1.0'
}

# up to date list can be obtained from:
# https://realtimebillionaires.de/api?request=profile%2F_index
try:
    with open('billionaires.json', 'r', encoding='utf-8') as f:
        billionaire_jsons = json.load(f)
        print(list(billionaire_jsons.items())[0])
        # Extract just the names from the billionaire data
        billionaires = []
        for entry in billionaire_jsons.values():
            try:
                billionaires.append(entry['name'])
            except:
                print(f"Error loading billionaire: {entry}")
                raise
except Exception as e:
    print(f"Error loading billionaires from JSON: {e}")


    raise



def get_headshot(name):
    """
    Query Wikipedia's API for the headshot (thumbnail) image of a given person.
    Returns the image URL if found, or None otherwise.
    """
    params = {
        "action": "query",
        "titles": name,
        "prop": "pageimages",
        "pithumbsize": 500,  # desired image width in pixels
        "format": "json"
    }
    try:
        response = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            # Check if a thumbnail exists in the result
            if "thumbnail" in page and "source" in page["thumbnail"]:
                return page["thumbnail"]["source"]
    except Exception as e:
        print(f"Error fetching image for {name}: {e}")
    return None

def download_image(url, path):
    """
    Download an image from the provided URL and save it at the given path.
    """
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded image to {path}")
    except Exception as e:
        print(f"Failed to download image from {url}. Reason: {e}")

def main():
    # Create folder for images if it doesn't exist
    images_dir = "./public/faces"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    for billionaire in billionaires:
        # Create a safe filename before checking if it exists
        filename = billionaire.lower().replace(" ", "-") + ".jpg"  # Using .jpg as default extension
        image_path = os.path.join(images_dir, filename)
        
        # Skip if image already exists
        if os.path.exists(image_path):
            print(f"Skipping {billionaire} - image already exists")
            continue
            
        print(f"Processing: {billionaire}")
        image_url = get_headshot(billionaire)
        if image_url:
            print(f"Found image URL: {image_url}")
            download_image(image_url, image_path)
        else:
            print(f"No image found for {billionaire}")
        # Wait a bit to be polite to the API server
        time.sleep(1)

if __name__ == "__main__":
    main()
