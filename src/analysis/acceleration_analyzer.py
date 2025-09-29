"""
Simplified acceleration analyzer - just the basics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from ..extract.data_loader import load_session_data
from .lap_analyzer import analyze_session_laps
from ..config.config_analysis import ENGINE_SPECS

def analyze_power_simple():
    """
    Simple power analysis: just F=ma, no complicated stuff
    """
    
    print("Simple Power Analysis")
    print("="*40)
    
    # Load session data
    session_data = load_session_data()
    if not session_data:
        print("Failed to load session data")
        return None
    
    # Get fastest lap
    laps, fastest_lap_data = analyze_session_laps(session_data)
    if not fastest_lap_data:
        print("No fastest lap data")
        return None
    
    lap_info = fastest_lap_data['lap_info']
    print(f"Fastest lap: #{lap_info.lap_number} ({lap_info.lap_time:.2f}s)")
    
    # Get data
    time = fastest_lap_data['time']
    speed_ms = fastest_lap_data.get('speed_ms', np.zeros(len(time)))
    rpm = fastest_lap_data['rpm']
    
    # Calculate acceleration
    dt = np.diff(time)
    dt[dt < 0.001] = 0.001
    dv = np.diff(speed_ms)
    accel = dv / dt
    
    # Smooth it
    if len(accel) > 10:
        window = min(11, len(accel) // 10)
        accel = signal.savgol_filter(accel, window, 3)
    
    # Vehicle mass
    mass_kg = 1565  # 3450 lbs
    
    # Simple power calculation: P = F * v = (m * a) * v
    power_data = []
    
    for i in range(len(accel)):
        if i >= len(speed_ms) - 1:
            break
            
        v = speed_ms[i+1]  # m/s
        a = accel[i]       # m/s²
        r = rpm[i+1]       # RPM
        
        if v > 5 and a > 0.1:  # Only when accelerating at decent speed
            force = mass_kg * a  # Newtons
            power_watts = force * v  # Watts
            power_hp = power_watts / 745.7  # HP
            
            power_data.append({
                'speed_mph': v * 2.237,
                'accel': a,
                'rpm': r,
                'power_hp': power_hp
            })
    
    if not power_data:
        print("No power data calculated")
        return None
    
    df = pd.DataFrame(power_data)
    
    print(f"Results:")
    print(f"  Data points: {len(df)}")
    print(f"  Power range: {df['power_hp'].min():.0f} - {df['power_hp'].max():.0f} HP")
    print(f"  RPM range: {df['rpm'].min():.0f} - {df['rpm'].max():.0f}")
    print(f"  Max speed: {df['speed_mph'].max():.1f} mph")
    
    # RPM analysis
    safe_limit = ENGINE_SPECS['safe_rpm_limit']
    over_limit = df[df['rpm'] > safe_limit]
    
    print(f"\nRPM Analysis:")
    print(f"  Safe limit: {safe_limit} RPM")
    print(f"  Max RPM: {df['rpm'].max():.0f}")
    
    if len(over_limit) > 0:
        print(f"  Points over limit: {len(over_limit)} ({len(over_limit)/len(df)*100:.1f}%)")
        print(f"  Speed range at risk: {over_limit['speed_mph'].min():.0f}-{over_limit['speed_mph'].max():.0f} mph")
        print(f"  RECOMMENDATION: Need taller gearing")
    else:
        print(f"  All RPM within safe limits")
    
    # Get GPS coordinates for track map
    lat = fastest_lap_data['latitude']
    lon = fastest_lap_data['longitude']
    
    # Match power data indices to GPS coordinates
    power_indices = []
    for i in range(len(accel)):
        if i+1 < len(speed_ms) and speed_ms[i+1] > 5 and accel[i] > 0.1:
            power_indices.append(i+1)
    
    # Create 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Track map with power zones
    if len(power_indices) > 0:
        # Get power values for color mapping
        power_values = df['power_hp'].values
        
        # Create power zones
        low_power = power_values < 150
        med_power = (power_values >= 150) & (power_values < 250)
        high_power = power_values >= 250
        
        # Plot track outline (all points)
        ax1.plot(lon, lat, 'lightgray', linewidth=1, alpha=0.5, label='Track')
        
        # Plot power zones
        if np.sum(low_power) > 0:
            indices = np.array(power_indices)[low_power]
            ax1.scatter(lon[indices], lat[indices], c='green', s=12, alpha=0.7, 
                       label=f'Low Power <150 HP ({np.sum(low_power)} pts)')
        
        if np.sum(med_power) > 0:
            indices = np.array(power_indices)[med_power]
            ax1.scatter(lon[indices], lat[indices], c='orange', s=12, alpha=0.7,
                       label=f'Med Power 150-250 HP ({np.sum(med_power)} pts)')
        
        if np.sum(high_power) > 0:
            indices = np.array(power_indices)[high_power]
            ax1.scatter(lon[indices], lat[indices], c='red', s=12, alpha=0.7,
                       label=f'High Power >250 HP ({np.sum(high_power)} pts)')
        
        # Mark start/finish
        ax1.plot(lon[0], lat[0], 'ko', markersize=8, markeredgewidth=2, 
                markeredgecolor='white', label='Start/Finish')
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Track Map - Power Application')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
    else:
        ax1.text(0.5, 0.5, 'No power data for mapping', transform=ax1.transAxes, 
                ha='center', va='center')
        ax1.set_title('Track Map - No Data')
    
    # 2. Power vs Speed
    ax2.scatter(df['speed_mph'], df['power_hp'], alpha=0.6, s=8)
    ax2.set_xlabel('Speed (mph)')
    ax2.set_ylabel('Power (HP)')
    ax2.set_title('Power vs Speed')
    ax2.grid(True, alpha=0.3)
    
    # 3. RPM vs Speed with safe limit
    ax3.scatter(df['speed_mph'], df['rpm'], alpha=0.6, s=8)
    ax3.axhline(y=safe_limit, color='red', linestyle='--', label=f'Safe Limit ({safe_limit})')
    ax3.set_xlabel('Speed (mph)')
    ax3.set_ylabel('RPM')
    ax3.set_title('RPM vs Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_power_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save data
    df.to_csv('simple_power_results.csv', index=False)
    
    return df

if __name__ == "__main__":
    analyze_power_simple()
