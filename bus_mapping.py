import requests
import json
import time
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("LTA_ACCOUNT_ID")
BASE_URL_STOPS = "https://datamall2.mytransport.sg/ltaodataservice/BusStops"
BASE_URL_ROUTES = "https://datamall2.mytransport.sg/ltaodataservice/BusRoutes"


def fetch_paginated_data(base_url, api_key, data_type):
    """
    Fetches paginated data from LTA DataMall API

    Args:
        base_url (str): API endpoint URL
        api_key (str): LTA DataMall API key
        data_type (str): Type of data being fetched (for logging)

    Returns:
        list: All records from the API
    """
    page_size = 500
    skip = 0
    all_data = []
    has_more_data = True
    page_count = 0

    headers = {"Accept": "application/json", "AccountKey": api_key}

    print(f"Fetching {data_type}...\n")

    try:
        while has_more_data:
            url = f"{base_url}?$skip={skip}"

            print(f"  Page {page_count + 1} (records {skip + 1}-{skip + page_size})...")

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            records = data.get("value", [])

            print(f"    ✓ Retrieved {len(records)} records")

            if len(records) > 0:
                all_data.extend(records)
                skip += page_size
                page_count += 1
                time.sleep(0.1)
            else:
                has_more_data = False

        print(f"  ✓ Total {data_type} retrieved: {len(all_data)}\n")
        return all_data

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error fetching {data_type}: {e}")
        raise


def aggregate_bus_services_by_stop(api_key, output_file="bus_stops_with_services.json"):
    """
    Fetches bus stops and routes, then aggregates bus services by stop code

    Args:
        api_key (str): Your LTA DataMall API key
        output_file (str): Path to output JSON file

    Returns:
        dict: Bus stops with their respective bus services
    """
    base_url_stops = BASE_URL_STOPS
    base_url_routes = BASE_URL_ROUTES

    print("=" * 60)
    print("LTA Bus Stops with Services Aggregator")
    print("=" * 60 + "\n")

    # Step 1: Fetch all bus stops
    print("[1/3] Fetching Bus Stops")
    print("-" * 60)
    bus_stops = fetch_paginated_data(base_url_stops, api_key, "bus stops")

    # Step 2: Fetch all bus routes
    print("[2/3] Fetching Bus Routes")
    print("-" * 60)
    bus_routes = fetch_paginated_data(base_url_routes, api_key, "bus routes")

    # Step 3: Aggregate bus services by stop code
    print("[3/3] Aggregating Bus Services by Stop Code")
    print("-" * 60)

    # Create a mapping of BusStopCode -> set of ServiceNo
    stop_services = defaultdict(set)

    for route in bus_routes:
        bus_stop_code = route.get("BusStopCode")
        service_no = route.get("ServiceNo")

        if bus_stop_code and service_no:
            stop_services[bus_stop_code].add(service_no)

    print(f"  ✓ Processed {len(bus_routes)} route records")
    print(f"  ✓ Found services for {len(stop_services)} bus stops\n")

    # Step 4: Combine bus stop info with service numbers
    bus_stops_with_services = []
    stops_with_no_services = 0

    for stop in bus_stops:
        bus_stop_code = stop.get("BusStopCode")
        services = sorted(list(stop_services.get(bus_stop_code, [])))

        if not services:
            stops_with_no_services += 1

        bus_stop_with_services = {
            "BusStopCode": bus_stop_code,
            "RoadName": stop.get("RoadName"),
            "Description": stop.get("Description"),
            "Latitude": stop.get("Latitude"),
            "Longitude": stop.get("Longitude"),
            "BusServiceNumbers": services,
        }

        bus_stops_with_services.append(bus_stop_with_services)

    # Create final output
    output = {
        "metadata": {
            "totalBusStops": len(bus_stops_with_services),
            "totalBusRoutes": len(bus_routes),
            "stopsWithServices": len(bus_stops_with_services) - stops_with_no_services,
            "stopsWithNoServices": stops_with_no_services,
            "timestamp": datetime.now().isoformat(),
            "source": "LTA DataMall API",
        },
        "busStops": bus_stops_with_services,
    }

    print("Writing to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total bus stops: {len(bus_stops_with_services)}")
    print(
        f"Stops with services: {len(bus_stops_with_services) - stops_with_no_services}"
    )
    print(f"Stops without services: {stops_with_no_services}")
    print(f"Total bus routes processed: {len(bus_routes)}")
    print(f"Output file: {output_file}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    try:
        result = aggregate_bus_services_by_stop(API_KEY, "bus_stops_with_services.json")

        print("\nSample bus stop with services:")
        sample = result["busStops"][0]
        print(json.dumps(sample, indent=2))

    except Exception as e:
        print(f"\n✗ Failed to aggregate bus data: {e}")
        exit(1)
