from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim

app = Flask(__name__)

@app.route('/location', methods=['POST'])
def track_location():
    data = request.json
    
    # Extract latitude and longitude from incoming data
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    
    # Use Geopy to get the address from the coordinates
    geolocator = Nominatim(user_agent="location_tracker")
    location = geolocator.reverse(f"{latitude}, {longitude}")
    
    # Return the address to the client
    return jsonify({'address': location.address})

if __name__ == '__main__':
    app.run(debug=True)
