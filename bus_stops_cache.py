import requests
import json
import time
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("LTA_ACCOUNT_ID")
BASE_STOPS_URL = "https://datamall2.mytransport.sg/ltaodataservice/BusStops"


def get_all_bus_stops(api_key, output_file="bus_stops.json"):
    """
    Fetches all bus stops from LTA DataMall API with pagination

    Args:
        api_key (str): Your LTA DataMall API key
        output_file (str): Path to output JSON file (default: bus_stops.json)

    Returns:
        dict: Combined results with all bus stops
    """
    base_url = BASE_STOPS_URL
    page_size = 500  # LTA API returns max 500 records per call
    skip = 0
    all_bus_stops = []
    has_more_data = True
    page_count = 0

    headers = {"Accept": "application/json", "AccountKey": api_key}

    print("Starting to fetch bus stops data...\n")

    try:
        while has_more_data:
            url = f"{base_url}?$skip={skip}"

            print(
                f"Fetching page {page_count + 1} (records {skip + 1}-{skip + page_size})..."
            )

            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for bad status codes

            data = response.json()
            bus_stops = data.get("value", [])

            print(f"  ✓ Retrieved {len(bus_stops)} records")

            if len(bus_stops) > 0:
                all_bus_stops.extend(bus_stops)
                skip += page_size
                page_count += 1

                # Add a small delay to avoid hitting rate limits
                time.sleep(0.1)
            else:
                has_more_data = False

        # Create the final output object
        output = {
            "metadata": {
                "totalRecords": len(all_bus_stops),
                "pagesRetrieved": page_count,
                "timestamp": datetime.now().isoformat(),
                "source": "LTA DataMall API",
            },
            "busStops": all_bus_stops,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Success! Total bus stops retrieved: {len(all_bus_stops)}")
        print(f"✓ Data saved to: {output_file}")

        return output

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error fetching bus stops: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    try:
        result = get_all_bus_stops(API_KEY, "bus_stops.json")
        print("\nSample bus stop:")
        print(json.dumps(result["busStops"][0], indent=2))
    except Exception as e:
        print(f"Failed to fetch bus stops: {e}")
        exit(1)
