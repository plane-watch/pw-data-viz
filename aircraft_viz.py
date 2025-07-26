#!/usr/bin/env python3
"""
Aircraft Path Visualization Module
A reusable module for creating aircraft flight path visualizations from ClickHouse CSV data.
Supports gzipped CSV files (.csv.gz) as input.
"""

import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import sys
import os
from datetime import datetime, timedelta
import io
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Try to import cartopy, fallback to basic matplotlib if not available
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    print("Warning: cartopy not available, using basic matplotlib")
    HAS_CARTOPY = False


class AircraftPathVisualizer:
    """
    A class for creating aircraft flight path visualizations with configurable styling and output options.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualizer with optional configuration.
        
        Args:
            config: Dictionary containing visualization configuration options
        """
        self.config = config or {}
        self.df = None
        self.flight_paths = {}
        self.bounds = None
        self.fig = None
        self.ax = None
        
        # Default configuration
        self.default_config = {
            'figure_size': (20, 15),
            'figure_size_large': (30, 20),
            'dpi': 300,
            'line_width': 0.8,
            'alpha_base': 0.3,
            'alpha_multiplier': 0.7,
            'grid_size': 200,
            'color_scheme': 'altitude',  # 'altitude', 'density', 'speed'
            'background_color': 'black',
            'use_cartopy': HAS_CARTOPY,
            'max_time_gap_minutes': 5  # Maximum time gap before splitting paths
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Altitude-based color scheme
        self.altitude_colors = {
            'ground': '#FFFFFF',      # White for ground/taxi
            'low': '#FFFF00',         # Yellow for low altitude
            'medium_low': '#FF8800',  # Orange for medium-low
            'medium': '#00FF00',      # Green for medium
            'high': '#00AAFF',        # Light blue for high
            'very_high': '#0066FF'    # Bright blue for very high
        }
        
        # Altitude thresholds (in feet)
        self.altitude_thresholds = [500, 5000, 15000, 25000, 35000]
        
    def load_data(self, csv_gz_path: str) -> pd.DataFrame:
        """
        Load CSV data from gzipped CSV file.
        
        Args:
            csv_gz_path: Path to the .csv.gz file containing CSV data
            
        Returns:
            Loaded DataFrame
        """
        if not csv_gz_path.endswith('.csv.gz'):
            raise ValueError("Input file must be a gzipped CSV file with .csv.gz extension")
            
        print(f"Loading data from {csv_gz_path}...")
        
        try:
            # Read gzipped CSV directly with pandas
            df = pd.read_csv(csv_gz_path, compression='gzip', header=None)
            print(f"Loaded {len(df)} records from {os.path.basename(csv_gz_path)}")
            
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {str(e)}")
            
        # Parse the data
        self.df = self._parse_aircraft_data(df)
        return self.df
    
    def _parse_aircraft_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the CSV data and extract relevant columns."""
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
        
        # Filter out records without location data
        df = df[df['HasLocation'] == True].copy()
        
        # Convert timestamp columns
        df['TrackedSince'] = pd.to_datetime(df['TrackedSince'])
        df['LastMsg'] = pd.to_datetime(df['LastMsg'])
        
        # Ensure altitude is numeric
        df['Altitude'] = pd.to_numeric(df['Altitude'], errors='coerce')
        
        print(f"Filtered to {len(df)} records with location data")
        return df
    
    def process_flight_paths(self) -> Dict:
        """
        Group aircraft positions by ICAO to create flight paths, splitting on time gaps.
        
        Returns:
            Dictionary of flight path segments keyed by ICAO code
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        print("Creating flight paths with time gap detection...")
        
        flight_paths = defaultdict(list)
        max_gap = timedelta(minutes=self.config['max_time_gap_minutes'])
        total_segments = 0
        
        # Group by aircraft ICAO code
        for icao, group in self.df.groupby('Icao'):
            # Sort by timestamp to create ordered path
            group_sorted = group.sort_values('LastMsg')
            
            # Extract coordinates and metadata
            all_coords = []
            for _, row in group_sorted.iterrows():
                all_coords.append({
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude']),
                    'timestamp': row['LastMsg'],
                    'altitude': row['Altitude'] if pd.notna(row['Altitude']) else 0,
                    'velocity': row['Velocity'] if pd.notna(row['Velocity']) else 0,
                    'on_ground': row['OnGround'],
                    'callsign': row['CallSign']
                })
            
            # Split into segments based on time gaps
            if len(all_coords) < 2:
                continue
                
            current_segment = [all_coords[0]]
            
            for i in range(1, len(all_coords)):
                current_coord = all_coords[i]
                previous_coord = all_coords[i-1]
                
                # Check time gap between consecutive points
                time_diff = current_coord['timestamp'] - previous_coord['timestamp']
                
                if time_diff <= max_gap:
                    # Continue current segment
                    current_segment.append(current_coord)
                else:
                    # Time gap detected - save current segment and start new one
                    if len(current_segment) >= 2:
                        flight_paths[icao].append(current_segment)
                        total_segments += 1
                    
                    # Start new segment
                    current_segment = [current_coord]
            
            # Add the final segment if it has enough points
            if len(current_segment) >= 2:
                flight_paths[icao].append(current_segment)
                total_segments += 1
        
        print(f"Created {total_segments} flight path segments from {len(flight_paths)} aircraft")
        print(f"Time gap threshold: {self.config['max_time_gap_minutes']} minutes")
        self.flight_paths = flight_paths
        return flight_paths
    
    def _calculate_map_bounds(self) -> Dict:
        """Calculate the geographic bounds for the map."""
        if not self.flight_paths:
            raise ValueError("No flight paths available. Call process_flight_paths() first.")
            
        all_lats = []
        all_lons = []
        
        # Handle segmented paths - each ICAO now has a list of segments
        for segments in self.flight_paths.values():
            for segment in segments:
                for point in segment:
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
        
        self.bounds = bounds
        return bounds
    
    def _get_altitude_color(self, altitude: float) -> str:
        """
        Get color based on altitude using smooth interpolation.
        
        Args:
            altitude: Altitude in feet
            
        Returns:
            Hex color string
        """
        if altitude <= self.altitude_thresholds[0]:  # Ground/taxi
            return self.altitude_colors['ground']
        elif altitude <= self.altitude_thresholds[1]:  # Low
            return self.altitude_colors['low']
        elif altitude <= self.altitude_thresholds[2]:  # Medium-low
            return self.altitude_colors['medium_low']
        elif altitude <= self.altitude_thresholds[3]:  # Medium
            return self.altitude_colors['medium']
        elif altitude <= self.altitude_thresholds[4]:  # High
            return self.altitude_colors['high']
        else:  # Very high
            return self.altitude_colors['very_high']
    
    def _create_altitude_colormap(self) -> LinearSegmentedColormap:
        """Create a smooth colormap for altitude visualization."""
        colors = list(self.altitude_colors.values())
        n_bins = len(colors)
        cmap = LinearSegmentedColormap.from_list(
            'altitude', colors, N=n_bins
        )
        return cmap
    
    def set_color_scheme(self, scheme: str = 'altitude'):
        """
        Set the color scheme for visualization.
        
        Args:
            scheme: Color scheme ('altitude', 'density', 'speed')
        """
        self.config['color_scheme'] = scheme
    
    def create_visualization(self, size: str = 'normal') -> Tuple:
        """
        Create the main visualization figure and axes.
        
        Args:
            size: Figure size ('normal' or 'large')
            
        Returns:
            Tuple of (figure, axes)
        """
        if not self.flight_paths:
            raise ValueError("No flight paths available. Call process_flight_paths() first.")
            
        if not self.bounds:
            self._calculate_map_bounds()
            
        print("Creating visualization...")
        
        # Set up the figure with dark theme
        plt.style.use('dark_background')
        
        # Choose figure size
        fig_size = (self.config['figure_size_large'] if size == 'large' 
                   else self.config['figure_size'])
        
        if self.config['use_cartopy'] and HAS_CARTOPY:
            # Use cartopy for proper map projection
            self.fig = plt.figure(figsize=fig_size, facecolor=self.config['background_color'])
            self.ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Set map extent
            self.ax.set_extent([self.bounds['min_lon'], self.bounds['max_lon'], 
                              self.bounds['min_lat'], self.bounds['max_lat']], 
                             crs=ccrs.PlateCarree())
            
            # Add map features with minimal styling
            self.ax.add_feature(cfeature.COASTLINE, color='#333333', linewidth=0.5)
            self.ax.add_feature(cfeature.BORDERS, color='#333333', linewidth=0.3)
            self.ax.add_feature(cfeature.OCEAN, color='#000000')
            self.ax.add_feature(cfeature.LAND, color='#111111')
            
        else:
            # Fallback to basic matplotlib
            self.fig, self.ax = plt.subplots(figsize=fig_size, 
                                           facecolor=self.config['background_color'])
            self.ax.set_xlim(self.bounds['min_lon'], self.bounds['max_lon'])
            self.ax.set_ylim(self.bounds['min_lat'], self.bounds['max_lat'])
            self.ax.set_facecolor(self.config['background_color'])
            self.ax.set_aspect('equal')
        
        # Remove axes and make it clean
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        
        return self.fig, self.ax
    
    def draw_flight_paths(self) -> None:
        """Draw the flight paths with the configured color scheme."""
        if not self.fig or not self.ax:
            raise ValueError("Visualization not created. Call create_visualization() first.")
            
        print("Drawing flight paths...")
        
        if self.config['color_scheme'] == 'altitude':
            self._draw_altitude_based_paths()
        else:
            # For now, default to altitude-based coloring
            self._draw_altitude_based_paths()
    
    def _draw_altitude_based_paths(self) -> None:
        """Draw flight paths colored by altitude."""
        segment_count = 0
        
        # Create colormap for smooth altitude transitions
        altitude_cmap = self._create_altitude_colormap()
        
        # Handle segmented paths - each ICAO now has a list of segments
        for icao, segments in self.flight_paths.items():
            for segment in segments:
                if len(segment) < 2:
                    continue
                    
                # Extract path coordinates and altitudes for this segment
                lats = [p['lat'] for p in segment]
                lons = [p['lon'] for p in segment]
                altitudes = [p['altitude'] for p in segment]
                
                # Calculate average altitude for the segment
                avg_altitude = np.mean([alt for alt in altitudes if alt > 0])
                if np.isnan(avg_altitude):
                    avg_altitude = 0
                
                # Get color based on average altitude
                color = self._get_altitude_color(avg_altitude)
                
                # Calculate transparency based on altitude
                # Higher altitudes get slightly more opacity
                if avg_altitude > 30000:
                    alpha = 0.8
                elif avg_altitude > 20000:
                    alpha = 0.6
                elif avg_altitude > 10000:
                    alpha = 0.5
                elif avg_altitude > 1000:
                    alpha = 0.4
                else:
                    alpha = 0.3
                
                # Draw the segment
                if self.config['use_cartopy'] and HAS_CARTOPY:
                    self.ax.plot(lons, lats, color=color, alpha=alpha, 
                               linewidth=self.config['line_width'], 
                               transform=ccrs.PlateCarree())
                else:
                    self.ax.plot(lons, lats, color=color, alpha=alpha, 
                               linewidth=self.config['line_width'])
                
                segment_count += 1
                
                # Progress indicator for large datasets
                if segment_count % 1000 == 0:
                    print(f"Drawn {segment_count} path segments...")
        
        print(f"Drew {segment_count} flight path segments using altitude-based coloring")
    
    def save_outputs(self, base_filename: str = 'aircraft_paths', 
                    formats: List[str] = ['png'], 
                    dpi: int = 300,
                    high_res: bool = False) -> List[str]:
        """
        Save the visualization in multiple formats.
        
        Args:
            base_filename: Base filename without extension
            formats: List of output formats ('png', 'svg', 'pdf')
            dpi: DPI for raster formats
            high_res: Whether to use high resolution settings
            
        Returns:
            List of created file paths
        """
        if not self.fig:
            raise ValueError("No figure to save. Call create_visualization() and draw_flight_paths() first.")
            
        saved_files = []
        
        # Set DPI based on high_res flag
        if high_res:
            dpi = max(dpi, 600)  # Minimum 600 DPI for high-res
        
        plt.tight_layout()
        
        for fmt in formats:
            filename = f"{base_filename}.{fmt}"
            
            print(f"Saving {fmt.upper()} to {filename}...")
            
            if fmt == 'png':
                self.fig.savefig(filename, 
                               dpi=dpi, 
                               bbox_inches='tight', 
                               facecolor=self.config['background_color'], 
                               edgecolor='none',
                               pad_inches=0.1,
                               format='png')
            
            elif fmt == 'svg':
                self.fig.savefig(filename, 
                               bbox_inches='tight', 
                               facecolor=self.config['background_color'], 
                               edgecolor='none',
                               pad_inches=0.1,
                               format='svg')
            
            elif fmt == 'pdf':
                self.fig.savefig(filename, 
                               bbox_inches='tight', 
                               facecolor=self.config['background_color'], 
                               edgecolor='none',
                               pad_inches=0.1,
                               format='pdf')
            
            else:
                print(f"Warning: Unsupported format '{fmt}', skipping...")
                continue
                
            saved_files.append(filename)
            print(f"✓ Saved {filename}")
        
        return saved_files
    
    def generate_visualization(self, csv_gz_path: str, 
                             output_base: str = 'aircraft_paths',
                             formats: List[str] = ['png'],
                             dpi: int = 300,
                             size: str = 'normal',
                             high_res: bool = False) -> List[str]:
        """
        Complete workflow: load data, process, visualize, and save.
        
        Args:
            csv_gz_path: Path to input .csv.gz file
            output_base: Base filename for outputs
            formats: List of output formats
            dpi: DPI for raster formats
            size: Figure size ('normal' or 'large')
            high_res: Enable high resolution output
            
        Returns:
            List of created file paths
        """
        print("Starting aircraft path visualization workflow...")
        
        # Load and process data
        self.load_data(csv_gz_path)
        self.process_flight_paths()
        
        # Create visualization
        self.create_visualization(size=size)
        self.draw_flight_paths()
        
        # Save outputs
        saved_files = self.save_outputs(output_base, formats, dpi, high_res)
        
        # Show statistics
        total_segments = sum(len(segments) for segments in self.flight_paths.values())
        total_points = sum(len(segment) for segments in self.flight_paths.values() for segment in segments)
        print(f"\n✓ Visualization completed successfully!")
        print(f"✓ Processed {len(self.flight_paths)} aircraft with {total_segments} path segments")
        print(f"✓ Total data points: {total_points:,}")
        print(f"✓ Map covers {self.bounds['min_lat']:.3f}° to {self.bounds['max_lat']:.3f}° latitude")
        print(f"✓ Map covers {self.bounds['min_lon']:.3f}° to {self.bounds['max_lon']:.3f}° longitude")
        print(f"✓ Files created: {', '.join(saved_files)}")
        
        return saved_files
