import csv
import json
import os
import sys

# Base Implementation for Plant Hardiness Zone Tool
# This tool:
# 1. Accepts an optional ZIP code as input.
# 2. Returns the general plant hardiness information and the zone map HTML.
# 3. If a ZIP code is provided and found in the datasets, returns the zone data.
# 4. If a ZIP code is provided but not found, notifies that no data is available for that ZIP code.

# CSV files containing ZIP-to-zone mappings:
CSV_FILES = [
    "phzm_ak_zipcode_2023.csv",
    "phzm_hi_zipcode_2023.csv",
    "phzm_pr_zipcode_2023.csv",
    "phzm_us_zipcode_2023.csv"
]

# The map HTML file:
map_html_raw = """<!-- Map Container -->
<div class="field field--name-field-arcgis-iframe field--type-iframe field--label-visually_hidden" style="display: flex; justify-content: center; align-items: center; padding: 20px;">
    <div class="field--label sr-only">ArcGIS iFrame</div>
    <div class="field--item">
      <div style="width: 800px; height: 800px;">
        <style type="text/css">
          iframe#iframe-field_arcgis_iframe-1 {
            border-width: 0;
            overflow: scroll;
            width: 100%;
            height: 100%;
          }
        </style>
        <iframe 
                name="iframe-field_arcgis_iframe-1" 
                id="iframe-field_arcgis_iframe-1" 
                title="USDA Plant Hardiness Zone Map"
                allow="accelerometer;autoplay;camera;encrypted-media;geolocation;gyroscope;microphone;payment;picture-in-picture" 
                allowfullscreen="allowfullscreen" 
                src="https://experience.arcgis.com/experience/b1d1fd9b284e46dcaa43959bec439b44/">
          Your browser does not support iframes, but you can visit 
          <a href="https://experience.arcgis.com/experience/b1d1fd9b284e46dcaa43959bec439b44/"></a>
        </iframe>
      </div>
    </div>
  </div>"""

# General info text (from the prompt, included verbatim):
GENERAL_INFO = """How are the zone numbers and colors interpreted? What do they mean?
The Plant Hardiness Zone Map (PHZM) is based on the average annual extreme minimum winter temperature, displayed as 10-degree F zones ranging from zone 1 (coldest) to zone 13 (warmest). Each zone is divided into half zones designated as ‘a’ and ‘b’. For example, 7a and 7b are 5-degree F increments representing the colder and warmer halves of zone 7, respectively. These designations serve as convenient labels and shorthand for communicating and comparing the extreme winter temperatures within the United States and Puerto Rico. Zone numbers are typically listed with the descriptions of perennial plants in catalogs and other sales information produced by commercial nurseries, plant suppliers, etc.

The sequence of colors assigned to the zones mimics the chromatic spectrum produced by a prism (i.e., red, orange, yellow, green, blue, indigo, and violet), providing a graphical representation of the plant hardiness zones. The overlay of colored zones on the map provides a convenient tool for understanding and comparing plant cold-hardiness across the United States and can facilitate the selection of appropriate perennial plants based on their observed performance in other regions of the country.

How should I use the map when growing plants?
All Plant Hardiness Zone Maps (PHZM) should serve as general guides for growing perennial plants. They are based on the average lowest temperatures, not the lowest ever. Zones in this edition of the USDA PHZM are based on 1991-2020 weather data. This does not represent the coldest it has ever been or ever will be in an area, but it simply is the average lowest winter temperatures for a given location for this 30-year span (1991-2020).

Consequently, growing plants at the extreme range of the coldest zone where they are adapted means that they could experience a year with a rare, extreme cold snap that lasts just a day or two, and plants that have thrived happily for several years could be lost. Gardeners need to keep that in mind and understand that past weather records cannot provide a guaranteed forecast for future variations in weather. They should consult with other knowledgeable producers and gardeners (e.g., established nurseries or Master Gardeners) or extension services (see the links on the homepage) with extensive expertise with conditions at their locales.
Furthermore, gardeners should recognize that many other environmental factors, in addition to hardiness zones, contribute to the success or failure of plants. Wind, soil type, soil moisture, humidity, pollution, snow, and winter sunshine can greatly affect the survival of plants. Warm season heat and moisture balance are particularly important in this regard. The way plants are placed in the landscape, how they are planted, and their size and health can also influence their survival.

• Light: To thrive, plants need to be planted where they will receive the proper amount of light. For example, plants that require partial shade that are at the limits of hardiness in your area might be injured by too much sun during the winter because it might cause rapid changes in the plant’s internal temperature.

• Soil moisture: Plants have different requirements for soil moisture, and this might vary seasonally. Plants that might otherwise be hardy in your zone might be injured if soil moisture is too dry in late autumn, and they enter dormancy while suffering moisture stress.

• Temperature: Plants grow best within a range of optimal temperatures, both cold and hot. That range may be wide for some varieties and species but narrow for others.

• Duration of exposure to cold: Many plants that can survive a short period of exposure to cold may not tolerate longer periods of cold weather.

• Humidity: High relative humidity limits cold damage by reducing moisture loss from leaves, branches, and buds. Cold injury can be more severe if the humidity is low, especially for evergreens.

Gardeners or nursery growers interested in more detailed information, or examples for how the PHZM could be applied as a decision-making tool for planting, can consult a publication developed for the 2012 version of the PHZM:

Widrlechner, M.P., C. Daly, M. Keller, and K. Kaplan. 2012. Horticultural Applications of a Newly Revised USDA Plant Hardiness Zone Map. HortTechnology, 22: 6-19. Available at https://dr.lib.iastate.edu/entities/publication/d476e893-64d9-457c-91da-2eb3818fa961

Section 2: How to Use the Map
Section 3: What’s New
Section 4: Background for Map and its Use

If your hardiness zone has changed in this edition of the USDA Plant Hardiness Zone Map (PHZM), it does not mean you should start removing plants from your garden or change what you are growing. What has thrived in your yard will most likely continue to thrive.

Hardiness zones in this map are based on the average annual extreme minimum temperature during a 30-year period in the past, not the lowest temperature that has ever occurred in the past or might occur in the future. Gardeners should keep that in mind when selecting plants, especially if they choose to "push" their hardiness zone by growing plants not rated for their zone. In addition, although this edition of the USDA PHZM is drawn in the most detailed scale (1/2 mile square) to date, there could still be microclimates that are too small to show up on the map.

Microclimates, which are fine-scale climate variations, can be small heat islands—such as those caused by blacktop and concrete—or cool spots (frost pockets) caused by small hills and valleys. Individual gardens also may have very localized microclimates. Your entire yard could be somewhat warmer or cooler than the surrounding area because it is sheltered or exposed. You also could have pockets within your garden that are warmer or cooler than the general zone for your area or for the rest of your yard, such as a sheltered area in front of a south-facing wall or a low spot where cold air pools first. No hardiness zone map can take the place of the detailed knowledge that gardeners learn about their own gardens through hands-on experience.

Many species of perennial plants gradually acquire cold hardiness in the fall when they experience shorter days and cooler temperatures. This hardiness is normally lost gradually in late winter as temperatures warm and days become longer. A bout of extremely cold weather early in the fall might injure plants even though the temperatures may not reach the average lowest temperature for your zone. Similarly, exceptionally warm weather in midwinter followed by a sharp change to seasonably cold weather may cause injury to plants as well. Such factors could not be taken into account in the USDA PHZM.

All PHZMs should serve as general guides. They are based on the average lowest temperatures, not the lowest ever. Growing plants at the extreme range of the coldest zone where they are adapted means that they could experience a year with a rare, extreme cold snap that lasts just a day or two, and plants that have thrived happily for several years could be lost. Gardeners need to keep that in mind and understand that past weather records cannot provide a guaranteed forecast for future variation in weather.

Other Factors Affecting Plant Survival
Many other environmental factors, in addition to hardiness zones, contribute to the success or failure of plants. Wind, soil type, soil moisture, humidity, pollution, snow, and winter sunshine can greatly affect the survival of plants. The way plants are placed in the landscape, how they are planted, and their size and health might also influence their survival.

• Light: To thrive, plants need to be planted where they will receive the proper amount of light. For example, plants that require partial shade that are at the limits of hardiness in your area might be injured by too much sun during the winter because it might cause rapid changes in the plant’s internal temperature.

• Soil moisture: Plants have different requirements for soil moisture, and this might vary seasonally. Plants that might otherwise be hardy in your zone might be injured if soil moisture is too dry in late autumn and they enter dormancy while suffering moisture stress.

• Temperature: Plants grow best within a range of optimal temperatures, both cold and hot. That range may be wide for some varieties and species but narrow for others.

• Duration of exposure to cold: Many plants that can survive a short period of exposure to cold may not tolerate longer periods of cold weather.

• Humidity: High relative humidity limits cold damage by reducing moisture loss from leaves, branches, and buds. Cold injury can be more severe if the humidity is low, especially for evergreens.
"""

def load_map_html():
    """Load the plant hardiness zone map HTML content."""
    if not os.path.exists(MAP_HTML_FILE):
        return "<!-- Map file not found -->"
    with open(MAP_HTML_FILE, "r", encoding="utf-8") as f:
        return f.read()

def find_zip_data(zipcode: str):
    """Searches for the given zipcode in all CSV files and returns the zone info if found."""
    zipcode = zipcode.strip()
    for csv_file in CSV_FILES:
        if os.path.exists(csv_file):
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("zipcode") == zipcode:
                        return {
                            "zipcode": zipcode,
                            "zone": row.get("zone"),
                            "temp_range": row.get("trange"),
                            "zone_title": row.get("zonetitle")
                        }
    return None

def run_tool(zipcode: str = ""):
    """Run the tool given an optional zipcode."""
    map_html = map_html_raw

    if zipcode == "":
        # No ZIP provided: just return general info and map.
        result = {
            "general_info": GENERAL_INFO,
            "map_html": map_html,
            "message": "No specific ZIP code provided. Please use the map’s ZIP code search function."
        }
        return result
    else:
        # ZIP provided: look it up.
        zone_data = find_zip_data(zipcode)
        if zone_data:
            # Found the zone data for this ZIP code
            result = {
                "general_info": GENERAL_INFO,
                "map_html": map_html,
                "zone_data": zone_data
            }
            return result
        else:
            # Not found
            result = {
                "general_info": GENERAL_INFO,
                "map_html": map_html,
                "message": f"No data available for ZIP code {zipcode}."
            }
            return result

def main():
    # Local testing examples:
    # If run with a zip code argument: python toolname.py 99501
    # If run without arguments: python toolname.py

    zipcode = ""
    if len(sys.argv) > 1:
        zipcode = sys.argv[1]

    output = run_tool(zipcode)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
