import os
import csv
import math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    """Extracts EXIF data from an image."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            exif = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_info = {}
                    for gps_tag, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_info[gps_tag_name] = gps_value
                    exif[tag_name] = gps_info
                else:
                    exif[tag_name] = value
            return exif
        else:
            return None
    except (IOError, OSError, AttributeError) as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def convert_gps_to_decimal(gps_info, coord_tag, ref_tag):
    coord = gps_info.get(coord_tag)
    ref = gps_info.get(ref_tag)
    if not coord or not ref:
        return None
    # coord is (IFDRational, IFDRational, IFDRational)
    deg = coord[0].numerator / coord[0].denominator
    min_val = coord[1].numerator / coord[1].denominator
    sec = coord[2].numerator / coord[2].denominator
    decimal = deg + min_val / 60 + sec / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def convert_exif_datetime_to_iso(exif_time):
    if exif_time == "N/A":
        return None
    # exif_time like "2023:10:15 14:30:45"
    parts = exif_time.split()
    date = parts[0].replace(':', '-')
    time = parts[1]
    return f"{date}T{time}Z"  # Assuming UTC

# Ask for the input directory
input_dir = input("Enter the directory containing JPG files: ")

# Get list of JPG files
jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

# Process the images and store the EXIF data in a list
image_data = []
kml_data = []
for filename in jpg_files:
    image_path = os.path.join(input_dir, filename)
    exif_data = get_exif_data(image_path)
    if exif_data:
        image_info = {}
        image_info["Filename"] = filename
        try:
            image_info["Time"] = exif_data["DateTimeOriginal"]
        except KeyError:
            image_info["Time"] = "N/A"

        try:
            gps_info = exif_data["GPSInfo"]
            lat_decimal = convert_gps_to_decimal(gps_info, 'GPSLatitude', 'GPSLatitudeRef')
            lon_decimal = convert_gps_to_decimal(gps_info, 'GPSLongitude', 'GPSLongitudeRef')
            if lat_decimal is not None:
                image_info["Latitude"] = math.radians(lat_decimal)
            else:
                image_info["Latitude"] = "N/A"
            if lon_decimal is not None:
                image_info["Longitude"] = math.radians(lon_decimal)
            else:
                image_info["Longitude"] = "N/A"
            alt = gps_info.get("GPSAltitude")
            if alt:
                image_info["Altitude"] = alt.numerator / alt.denominator  # Convert IFDRational to float
            else:
                image_info["Altitude"] = "N/A"
            # Collect data for GPX and KML
            if lat_decimal is not None and lon_decimal is not None:
                alt_val = image_info["Altitude"] if image_info["Altitude"] != "N/A" else 0
                time_iso = convert_exif_datetime_to_iso(image_info["Time"])
                kml_data.append((lat_decimal, lon_decimal, alt_val, filename, time_iso))
        except (KeyError, AttributeError, IndexError):
            image_info["Latitude"] = "N/A"
            image_info["Longitude"] = "N/A"
            image_info["Altitude"] = "N/A"
        image_data.append(image_info)

# Ask for the output directory and filename
output_dir = input("Enter the output directory: ")
output_filename = input("Enter the desired filename for the CSV output: ")
output_path = os.path.join(output_dir, output_filename+".csv")

# Write the EXIF data to the CSV file
with open(output_path, 'w', newline='') as csvfile:
    fieldnames = ['Filename', 'Time', 'Latitude', 'Longitude', 'Altitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(image_data)

print(f"CSV file saved to {output_path}")

# Write the GPX file
if kml_data:
    gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="Photo EXIF Extractor" xmlns="http://www.topografix.com/GPX/1/1">\n'
    for lat, lon, alt, fname, time_iso in kml_data:
        gpx_content += f'<wpt lat="{lat}" lon="{lon}">\n<ele>{alt}</ele>\n<name>{fname}</name>\n'
        if time_iso:
            gpx_content += f'<time>{time_iso}</time>\n'
        gpx_content += '</wpt>\n'
    gpx_content += '</gpx>'
    gpx_path = output_path.replace('.csv', '.gpx')
    with open(gpx_path, 'w', encoding='utf-8') as f:
        f.write(gpx_content)
    print(f"GPX file saved to {gpx_path}")
else:
    print("No GPS data available for GPX file.")

# Write the KML file
if kml_data:
    kml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n'
    for lat, lon, alt, fname, time_iso in kml_data:
        kml_content += f'<Placemark>\n<name>{fname}</name>\n'
        if time_iso:
            kml_content += f'<TimeStamp><when>{time_iso}</when></TimeStamp>\n'
        kml_content += f'<Point>\n<coordinates>{lon},{lat},{alt}</coordinates>\n</Point>\n</Placemark>\n'
    kml_content += '</Document>\n</kml>'
    kml_path = output_path.replace('.csv', '.kml')
    with open(kml_path, 'w', encoding='utf-8') as f:
        f.write(kml_content)
    print(f"KML file saved to {kml_path}")
else:
    print("No GPS data available for KML file.")