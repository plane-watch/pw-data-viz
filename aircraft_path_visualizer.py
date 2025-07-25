#!/usr/bin/env python3
"""
Aircraft Path Visualizer
Creates a dark map visualization with aircraft flight paths from ClickHouse CSV data.
Usage: python aircraft_path_visualizer.py <input.tar.gz>
"""

import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import sys
import os
from datetime import datetime
import io

# Try to import cartopy, fallback to basic matplotlib if not available
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    print("Warning: cartopy not available, using basic matplotlib")
    HAS_CARTOPY = False

def extract_csv_from_tar(tar_path):
    """Extract CSV file from tar.gz archive"""
    print(f"Extracting data from {tar_path}...")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Find the CSV file in the archive
        csv_files = [member for member in tar.getmembers() if member.name.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV file found in archive")
        
        if len(csv_files) > 1:
            print(f"Multiple CSV files found, using: {csv_files[0].name}")
        
        # Extract and read the CSV
        csv_member = csv_files[0]
        extracted_file = tar.extractfile(csv_member)
        
        if extracted_file is None:
            raise ValueError(f"Could not extract {csv_member.name}")
        
        # Read CSV into pandas DataFrame
        csv_content = extracted_file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content), header=None)
        
        print(f"Loaded {len(df)} records from {csv_member.name}")
        return df

def parse_aircraft_data(df):
    """Parse the CSV data and extract relevant columns"""
    print("Parsing aircraft data...")
    
    # Define column names based on the schema provided
    column_names = [
        'Icao', 'Latitude', 'Longitude', 'Heading', 'Velocity', 'Altitude', 
        'VerticalRate', 'AltitudeUnits', 'CallSign', 'FlightStatus', 'OnGround',
        'Airframe', 'AirframeType', 'HasLocation', 'HasHeading', 'HasVerticalRate',
        'HasVelocity', 'SourceTags', 'Squawk', 'Special', 'TrackedSince', 'LastMsg',
        'FlagCode', 'Operator', 'RegisteredOwner', 'Registration', 'RouteCode',
        'Serial', 'TileLocation', 'TypeCode'
    ]
    
    # Assign column names
    df.columns = column_names[:len(df.columns)]
    
    # Convert timestamp columns
    df['TrackedSince'] = pd.to_datetime(df['TrackedSince'])
    df['LastMsg'] = pd.to_datetime(df['LastMsg'])
    
    print(f"Filtered to {len(df)} records with location data")
    return df

def create_flight_paths(df):
    """Group aircraft positions by ICAO to create flight paths"""
    print("Creating flight paths...")
    
    flight_paths = defaultdict(list)
    
    # Group by aircraft ICAO code
    for icao, group in df.groupby('Icao'):
        # Sort by timestamp to create ordered path
        group_sorted = group.sort_values('LastMsg')
        
        # Extract coordinates and timestamps
        coords = []
        for _, row in group_sorted.iterrows():
            coords.append({
                'lat': float(row['Latitude']),
                'lon': float(row['Longitude']),
                'timestamp': row['LastMsg'],
                'altitude': row['Altitude'],
                'on_ground': row['OnGround']
            })
        
        # Only include paths with multiple points
        if len(coords) > 1:
            flight_paths[icao] = coords
    
    print(f"Created {len(flight_paths)} flight paths")
    return flight_paths

def calculate_map_bounds(flight_paths):
    """Calculate the geographic bounds for the map"""
    all_lats = []
    all_lons = []
    
    for path in flight_paths.values():
        for point in path:
            all_lats.append(point['lat'])
            all_lons.append(point['lon'])
    
    if not all_lats:
        return None
    
    # Add some padding around the bounds
    lat_padding = (max(all_lats) - min(all_lats)) * 0.1
    lon_padding = (max(all_lons) - min(all_lons)) * 0.1
    
    bounds = {
        'min_lat': min(all_lats) - lat_padding,
        'max_lat': max(all_lats) + lat_padding,
        'min_lon': min(all_lons) - lon_padding,
        'max_lon': max(all_lons) + lon_padding
    }
    
    print(f"Map bounds: {bounds['min_lat']:.3f} to {bounds['max_lat']:.3f} lat, "
          f"{bounds['min_lon']:.3f} to {bounds['max_lon']:.3f} lon")
    
    return bounds

def create_density_grid(flight_paths, bounds, grid_size=200):
    """Create a density grid to show path frequency"""
    print("Calculating path density...")
    
    # Create coordinate arrays for the grid
    lat_range = np.linspace(bounds['min_lat'], bounds['max_lat'], grid_size)
    lon_range = np.linspace(bounds['min_lon'], bounds['max_lon'], grid_size)
    
    # Initialize density grid
    density_grid = np.zeros((grid_size, grid_size))
    
    # Calculate cell sizes
    lat_cell_size = (bounds['max_lat'] - bounds['min_lat']) / grid_size
    lon_cell_size = (bounds['max_lon'] - bounds['min_lon']) / grid_size
    
    # Populate density grid
    for path in flight_paths.values():
        for i in range(len(path) - 1):
            # Get path segment
            p1 = path[i]
            p2 = path[i + 1]
            
            # Interpolate points along the segment for better coverage
            num_interp = max(2, int(np.sqrt((p2['lat'] - p1['lat'])**2 + (p2['lon'] - p1['lon'])**2) * 1000))
            
            for t in np.linspace(0, 1, num_interp):
                lat = p1['lat'] + t * (p2['lat'] - p1['lat'])
                lon = p1['lon'] + t * (p2['lon'] - p1['lon'])
                
                # Convert to grid coordinates
                lat_idx = int((lat - bounds['min_lat']) / lat_cell_size)
                lon_idx = int((lon - bounds['min_lon']) / lon_cell_size)
                
                # Check bounds and increment density
                if 0 <= lat_idx < grid_size and 0 <= lon_idx < grid_size:
                    density_grid[lat_idx, lon_idx] += 1
    
    return density_grid, lat_range, lon_range

def create_visualization(flight_paths, bounds, output_path='aircraft_paths.png'):
    """Create the main visualization"""
    print("Creating visualization...")
    
    # Set up the figure with dark theme
    plt.style.use('dark_background')
    
    if HAS_CARTOPY:
        # Use cartopy for proper map projection
        fig = plt.figure(figsize=(20, 15), facecolor='black')
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_extent([bounds['min_lon'], bounds['max_lon'], 
                      bounds['min_lat'], bounds['max_lat']], 
                     crs=ccrs.PlateCarree())
        
        # Add map features with minimal styling
        ax.add_feature(cfeature.COASTLINE, color='#333333', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, color='#333333', linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, color='#000000')
        ax.add_feature(cfeature.LAND, color='#111111')
        
    else:
        # Fallback to basic matplotlib
        fig, ax = plt.subplots(figsize=(20, 15), facecolor='black')
        ax.set_xlim(bounds['min_lon'], bounds['max_lon'])
        ax.set_ylim(bounds['min_lat'], bounds['max_lat'])
        ax.set_facecolor('black')
        ax.set_aspect('equal')
    
    # Remove axes and make it clean
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig, ax

def draw_flight_paths(ax, flight_paths, bounds, use_cartopy=False):
    """Draw the flight paths with density-based coloring"""
    print("Drawing flight paths...")
    
    # Create density grid for transparency calculation
    density_grid, lat_range, lon_range = create_density_grid(flight_paths, bounds)
    max_density = np.max(density_grid)
    
    if max_density == 0:
        print("Warning: No density calculated")
        max_density = 1
    
    # Color schemes for different visualization options
    colors = ['#00FFFF', '#00FF88', '#88FF00', '#FFFF00', '#FF8800', '#FF0088']
    
    path_count = 0
    for icao, path in flight_paths.items():
        if len(path) < 2:
            continue
            
        # Extract path coordinates
        lats = [p['lat'] for p in path]
        lons = [p['lon'] for p in path]
        
        # Calculate average density for this path
        path_density = 0
        density_points = 0
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            
            # Sample density along the path segment
            for t in np.linspace(0, 1, 5):
                lat = p1['lat'] + t * (p2['lat'] - p1['lat'])
                lon = p1['lon'] + t * (p2['lon'] - p1['lon'])
                
                # Find nearest grid cell
                lat_idx = min(len(lat_range) - 1, max(0, int((lat - bounds['min_lat']) / 
                                                           (bounds['max_lat'] - bounds['min_lat']) * len(lat_range))))
                lon_idx = min(len(lon_range) - 1, max(0, int((lon - bounds['min_lon']) / 
                                                           (bounds['max_lon'] - bounds['min_lon']) * len(lon_range))))
                
                if 0 <= lat_idx < len(lat_range) and 0 <= lon_idx < len(lon_range):
                    path_density += density_grid[lat_idx, lon_idx]
                    density_points += 1
        
        # Calculate transparency based on density
        if density_points > 0:
            avg_density = path_density / density_points
            alpha = min(1.0, 0.1 + (avg_density / max_density) * 0.9)
        else:
            alpha = 0.1
        
        # Choose color based on path characteristics
        # Use cyan/purple for high-density paths, yellow/orange for medium, etc.
        if avg_density > max_density * 0.7:
            color = '#00FFFF'  # Bright cyan for highest density
        elif avg_density > max_density * 0.4:
            color = '#8800FF'  # Purple for high density
        elif avg_density > max_density * 0.2:
            color = '#FFFF00'  # Yellow for medium density
        else:
            color = '#FF8800'  # Orange for lower density
        
        # Draw the path
        if use_cartopy and HAS_CARTOPY:
            ax.plot(lons, lats, color=color, alpha=alpha, linewidth=0.8, 
                   transform=ccrs.PlateCarree())
        else:
            ax.plot(lons, lats, color=color, alpha=alpha, linewidth=0.8)
        
        path_count += 1
        
        # Progress indicator for large datasets
        if path_count % 1000 == 0:
            print(f"Drawn {path_count} paths...")
    
    print(f"Drew {path_count} flight paths")
    
    return ax

def main():
    """Main function to run the aircraft path visualizer"""
    if len(sys.argv) != 2:
        print("Usage: python aircraft_path_visualizer.py <input.tar.gz>")
        sys.exit(1)
    
    tar_path = sys.argv[1]
    
    if not os.path.exists(tar_path):
        print(f"Error: File {tar_path} not found")
        sys.exit(1)
    
    try:
        # Step 1: Extract and load data
        df = extract_csv_from_tar(tar_path)
        
        # Step 2: Parse aircraft data
        aircraft_data = parse_aircraft_data(df)
        
        # Step 3: Create flight paths
        flight_paths = create_flight_paths(aircraft_data)
        
        if not flight_paths:
            print("Error: No flight paths found in data")
            sys.exit(1)
        
        # Step 4: Calculate map bounds
        bounds = calculate_map_bounds(flight_paths)
        
        if bounds is None:
            print("Error: Could not calculate map bounds")
            sys.exit(1)
        
        # Step 5: Create visualization
        fig, ax = create_visualization(flight_paths, bounds)
        
        # Step 6: Draw flight paths
        ax = draw_flight_paths(ax, flight_paths, bounds, use_cartopy=HAS_CARTOPY)
        
        # Step 7: Save the image
        output_path = 'aircraft_paths_visualization.png'
        print(f"Saving visualization to {output_path}...")
        
        plt.tight_layout()
        plt.savefig(output_path, 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='black', 
                   edgecolor='none',
                   pad_inches=0.1)
        
        print(f"✓ Visualization saved successfully to {output_path}")
        print(f"✓ Processed {len(flight_paths)} flight paths")
        print(f"✓ Map covers {bounds['min_lat']:.3f}° to {bounds['max_lat']:.3f}° latitude")
        print(f"✓ Map covers {bounds['min_lon']:.3f}° to {bounds['max_lon']:.3f}° longitude")
        
        # Show some statistics
        total_points = sum(len(path) for path in flight_paths.values())
        print(f"✓ Total data points: {total_points:,}")
        
        # Optional: Show the plot (comment out for headless systems)
        # plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
