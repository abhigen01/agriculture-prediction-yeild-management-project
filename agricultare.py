import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
import os
import sys
import random

# ANSI escape codes for text formatting and color
BOLD = '\033[1m'
RESET = '\033[0m'
COLORS = [
    '\033[31m', # Red
    '\033[32m', # Green
    '\033[33m', # Yellow
    '\033[34m', # Blue
    '\033[35m', # Magenta
    '\033[36m', # Cyan
    '\033[91m', # Bright Red
    '\033[92m', # Bright Green
    '\033[93m', # Bright Yellow
    '\033[94m', # Bright Blue
    '\033[95m', # Bright Magenta
    '\033[96m', # Bright Cyan
]

# Expanded crop list
CROPS = ['wheat', 'rice', 'corn', 'barley', 'oats', 'sorghum', 'soybeans', 'potatoes', 'tomatoes', 'cotton']

# Expanded sample data (replace with a more comprehensive dataset for real-world use)
data = pd.DataFrame({
    'crop': CROPS * 30,
    'temp_min': np.random.uniform(10, 25, 300),
    'temp_max': np.random.uniform(20, 35, 300),
    'area': np.random.uniform(50, 500, 300),
    'humidity_min': np.random.uniform(40, 70, 300),
    'humidity_max': np.random.uniform(60, 90, 300),
    'rainfall_min': np.random.uniform(300, 1000, 300),
    'rainfall_max': np.random.uniform(500, 1500, 300),
    'soil_ph_min': np.random.uniform(5.5, 7.0, 300),
    'soil_ph_max': np.random.uniform(6.0, 7.5, 300)
})

# Calculate average values for model training
data['temp_avg'] = (data['temp_min'] + data['temp_max']) / 2
data['humidity_avg'] = (data['humidity_min'] + data['humidity_max']) / 2
data['rainfall_avg'] = (data['rainfall_min'] + data['rainfall_max']) / 2
data['soil_ph_avg'] = (data['soil_ph_min'] + data['soil_ph_max']) / 2

# Generate synthetic yield data based on optimal conditions for each crop
def generate_yield(row):
    base_yield = np.random.uniform(2000, 7000)
    if row['crop'] == 'wheat':
        return base_yield * (1 - abs(row['temp_avg'] - 20) / 20) * (1 - abs(row['humidity_avg'] - 60) / 60) * (row['rainfall_avg'] / 800) * (1 - abs(row['soil_ph_avg'] - 6.5) / 2)
    elif row['crop'] == 'rice':
        return base_yield * (1 - abs(row['temp_avg'] - 25) / 25) * (row['humidity_avg'] / 80) * (row['rainfall_avg'] / 1200) * (1 - abs(row['soil_ph_avg'] - 6.0) / 2)
    elif row['crop'] == 'corn':
        return base_yield * (1 - abs(row['temp_avg'] - 28) / 28) * (1 - abs(row['humidity_avg'] - 65) / 65) * (row['rainfall_avg'] / 700) * (1 - abs(row['soil_ph_avg'] - 6.8) / 2)
    else:  # For other crops, use a generic formula
        return base_yield * (1 - abs(row['temp_avg'] - 25) / 25) * (1 - abs(row['humidity_avg'] - 65) / 65) * (row['rainfall_avg'] / 1000) * (1 - abs(row['soil_ph_avg'] - 6.5) / 2)

data['yield'] = data.apply(generate_yield, axis=1)

# Train a Random Forest model
X = data[['temp_avg', 'area', 'humidity_avg', 'rainfall_avg', 'soil_ph_avg']]
y = data['yield']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

def predict_yield(crop, temp_min, temp_max, area, humidity_min, humidity_max, rainfall_min, rainfall_max, soil_ph_min, soil_ph_max):
    temp_avg = (temp_min + temp_max) / 2
    humidity_avg = (humidity_min + humidity_max) / 2
    rainfall_avg = (rainfall_min + rainfall_max) / 2
    soil_ph_avg = (soil_ph_min + soil_ph_max) / 2
    prediction = model.predict([[temp_avg, area, humidity_avg, rainfall_avg, soil_ph_avg]])[0]
    return max(0, prediction)  # Ensure non-negative yield

def suggest_measures(yield_value, crop, temp_min, temp_max, humidity_min, humidity_max, rainfall_min, rainfall_max, soil_ph_min, soil_ph_max):
    suggestions = []
    if yield_value < 3000:
        suggestions.append("Consider soil testing and appropriate fertilization.")
        suggestions.append("Implement improved irrigation techniques.")
        suggestions.append("Apply pest control measures.")
    elif yield_value < 5000:
        suggestions.append("Optimize nutrient management for better yields.")
        suggestions.append("Implement crop rotation to improve soil health.")
    else:
        suggestions.append("Maintain current practices and monitor for any changes.")

    # Crop-specific suggestions
    if crop in ['wheat', 'barley', 'oats']:
        if temp_max > 25:
            suggestions.append(f"Maximum temperature is high for {crop}. Consider heat-resistant varieties or adjust planting dates.")
        if humidity_max > 70:
            suggestions.append(f"High humidity may increase disease risk for {crop}. Implement fungicide treatments and improve air circulation.")
    elif crop in ['rice', 'sorghum']:
        if rainfall_min < 1000:
            suggestions.append(f"Minimum rainfall might be insufficient for {crop}. Consider supplemental irrigation or water-efficient varieties.")
        if soil_ph_min < 5.5 or soil_ph_max > 6.5:
            suggestions.append(f"Adjust soil pH to optimal range (5.5-6.5) for {crop} cultivation.")
    elif crop in ['corn', 'soybeans']:
        if temp_min < 20 or temp_max > 30:
            suggestions.append(f"Temperature range is suboptimal for {crop}. Adjust planting dates or consider different varieties.")
        if rainfall_min < 500 or rainfall_max > 800:
            suggestions.append(f"Rainfall range is not ideal for {crop}. Implement proper drainage or irrigation systems.")
    elif crop in ['potatoes', 'tomatoes']:
        if soil_ph_min < 6.0 or soil_ph_max > 6.8:
            suggestions.append(f"Adjust soil pH to optimal range (6.0-6.8) for {crop} cultivation.")
        if humidity_max > 80:
            suggestions.append(f"High humidity may increase disease risk for {crop}. Improve air circulation and consider fungicide treatments.")
    elif crop == 'cotton':
        if temp_min < 15 or temp_max > 35:
            suggestions.append("Temperature range is suboptimal for cotton. Adjust planting dates or consider different varieties.")
        if rainfall_min < 600 or rainfall_max > 1200:
            suggestions.append("Rainfall range is not ideal for cotton. Implement proper irrigation or drainage systems.")

    return suggestions

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def animate_text(text, is_title=False, rainbow=False):
    if is_title:
        text = f"{BOLD}{text}{RESET}"
    for i, char in enumerate(text):
        if rainbow:
            color = random.choice(COLORS)
            sys.stdout.write(f"{color}{char}{RESET}")
        else:
            sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.02)
    print()

def create_animated_graph(crop, temp_min, temp_max, area, humidity_min, humidity_max, rainfall_min, rainfall_max, soil_ph_min, soil_ph_max):
    initial_yield = predict_yield(crop, temp_min, temp_max, area, humidity_min, humidity_max, rainfall_min, rainfall_max, soil_ph_min, soil_ph_max)
    suggestions = suggest_measures(initial_yield, crop, temp_min, temp_max, humidity_min, humidity_max, rainfall_min, rainfall_max, soil_ph_min, soil_ph_max)

    # Simulate improved yield after implementing suggestions
    improved_yield = initial_yield * 1.2  # Assume 20% improvement

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Predicted Yield for {crop.capitalize()}', fontsize=16)

    # Set up both subplots
    for ax in (ax1, ax2):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(8000, improved_yield * 1.2))
        ax.set_xlabel('Days')
        ax.set_ylabel('Predicted Yield (kg/hectare)')
    
    ax1.set_title('Before Preventive Measures')
    ax2.set_title('After Preventive Measures')

    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)

    # Create a custom colormap for the RGB effect
    colors = ['red', 'green', 'blue']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('rgb', colors, N=n_bins)

    # Add attractive data points
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    season_days = [25, 50, 75, 100]
    season_yields_before = [initial_yield * 0.8, initial_yield * 1.0, initial_yield * 1.1, initial_yield * 0.9]
    season_yields_after = [improved_yield * 0.8, improved_yield * 1.0, improved_yield * 1.1, improved_yield * 0.9]

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        x = np.linspace(0, i, 100)
        y1 = initial_yield * (1 - np.exp(-x/30))
        y2 = improved_yield * (1 - np.exp(-x/30))
        line1.set_data(x, y1)
        line2.set_data(x, y2)
        line1.set_color(cmap(i/100))
        line2.set_color(cmap(i/100))

        # Add seasonal data points
        for j, (day, yield_before, yield_after) in enumerate(zip(season_days, season_yields_before, season_yields_after)):
            if i >= day:
                ax1.scatter(day, yield_before, c='red', s=100, zorder=5)
                ax2.scatter(day, yield_after, c='green', s=100, zorder=5)
                ax1.annotate(seasons[j], (day, yield_before), xytext=(5, 5), textcoords='offset points')
                ax2.annotate(seasons[j], (day, yield_after), xytext=(5, 5), textcoords='offset points')
        return line1, line2

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=False)
    plt.show()
    return initial_yield, improved_yield, suggestions

def get_float_input(prompt, min_value, max_value):
    while True:
        try:
            value = float(input(f"{random.choice(COLORS)}{prompt} {RESET}"))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    while True:
        clear_console()
        animate_text("Enhanced Agricultural Productivity Analyzer", is_title=True, rainbow=True)
        animate_text("Please enter the following information:", is_title=True, rainbow=True)

        crop = input(f"{random.choice(COLORS)}Crop name ({'/'.join(CROPS)}): {RESET}").lower()
        while crop not in CROPS:
            crop = input(f"{random.choice(COLORS)}Please enter a valid crop ({'/'.join(CROPS)}): {RESET}").lower()

        temp_min = get_float_input("Minimum Temperature (°C):", 0, 40)
        temp_max = get_float_input("Maximum Temperature (°C):", temp_min, 50)
        area = get_float_input("Land area (hectares):", 1, 1000)
        humidity_min = get_float_input("Minimum Humidity (%):", 0, 100)
        humidity_max = get_float_input("Maximum Humidity (%):", humidity_min, 100)
        rainfall_min = get_float_input("Minimum Annual rainfall (mm):", 0, 2000)
        rainfall_max = get_float_input("Maximum Annual rainfall (mm):", rainfall_min, 3000)
        soil_ph_min = get_float_input("Minimum Soil pH:", 0, 14)
        soil_ph_max = get_float_input("Maximum Soil pH:", soil_ph_min, 14)

        clear_console()
        animate_text("Analyzing data and generating predictions...", is_title=True, rainbow=True)
        time.sleep(1)

        initial_yield, improved_yield, suggestions = create_animated_graph(crop, temp_min, temp_max, area, humidity_min, humidity_max, rainfall_min, rainfall_max, soil_ph_min, soil_ph_max)

        clear_console()
        animate_text(f"Initial predicted yield for {crop}: {initial_yield:.2f} kg/hectare", is_title=True, rainbow=True)
        animate_text(f"Improved yield after measures: {improved_yield:.2f} kg/hectare", is_title=True, rainbow=True)
        animate_text(f"Potential improvement: {(improved_yield - initial_yield):.2f} kg/hectare ({((improved_yield / initial_yield) - 1) * 100:.2f}%)", is_title=True, rainbow=True)
        animate_text("\nSuggestions to improve productivity:", is_title=True, rainbow=True)

        for i, suggestion in enumerate(suggestions, 1):
            animate_text(f"{i}. {suggestion}", rainbow=True)

if __name__ == "__main__":
    main()