import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("LTA_ACCOUNT_ID")
BASE_ROUTES_URL = "https://datamall2.mytransport.sg/ltaodataservice/BusRoutes"


def get_all_bus_routes(api_key, output_file="bus_routes.json"):
    """
    Fetches all bus routes from LTA DataMall API with pagination

    Args:
        api_key (str): Your LTA DataMall API key
        output_file (str): Path to output JSON file (default: bus_routes.json)

    Returns:
        dict: Combined results with all bus routes
    """
    base_url = BASE_ROUTES_URL
    page_size = 500  # LTA API returns max 500 records per call
    skip = 0
    all_bus_routes = []
    has_more_data = True
    page_count = 0

    headers = {"Accept": "application/json", "AccountKey": api_key}

    print("Starting to fetch bus routes data...\n")

    try:
        while has_more_data:
            url = f"{base_url}?$skip={skip}"

            print(
                f"Fetching page {page_count + 1} (records {skip + 1}-{skip + page_size})..."
            )

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            bus_routes = data.get("value", [])

            print(f"  ✓ Retrieved {len(bus_routes)} records")

            if len(bus_routes) > 0:
                all_bus_routes.extend(bus_routes)
                skip += page_size
                page_count += 1

                time.sleep(0.1)
            else:
                has_more_data = False

        # Create the final output object
        output = {
            "metadata": {
                "totalRecords": len(all_bus_routes),
                "pagesRetrieved": page_count,
                "timestamp": datetime.now().isoformat(),
                "source": "LTA DataMall API",
            },
            "busRoutes": all_bus_routes,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Success! Total bus routes retrieved: {len(all_bus_routes)}")
        print(f"✓ Data saved to: {output_file}")

        return output

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error fetching bus routes: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    try:
        result = get_all_bus_routes(API_KEY, "bus_routes.json")
        print("\nSample bus route:")
        print(json.dumps(result["busRoutes"][0], indent=2))
    except Exception as e:
        print(f"Failed to fetch bus routes: {e}")
        exit(1)
