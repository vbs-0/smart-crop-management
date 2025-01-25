# fertilizer_recommendations.py

import pandas as pd

def get_fertilizer_recommendation(N, P, K, pH, crop=None):
    recommendations = []
    
    # N recommendations
    if N < 50:
        recommendations.append("""Nitrogen levels are low. Consider the following: 
         1. Apply nitrogen-rich fertilizers like urea or ammonium sulfate.
        2. Incorporate legumes in crop rotation to naturally fix nitrogen.
        3. Use organic compost or well-rotted manure to improve soil nitrogen content.
        """)
    elif N > 200:
        recommendations.append("""
        Nitrogen levels are high. Consider the following:
        1. Reduce or avoid nitrogen fertilizers temporarily.
        2. Plant nitrogen-hungry crops like corn or spinach.
        3. Add organic mulch with a high carbon-to-nitrogen ratio to balance soil.
        """)
    
    # P recommendations
    if P < 10:
        recommendations.append("""
        Phosphorus levels are low. Consider the following:
        1. Apply phosphate fertilizers or rock phosphate.
        2. Incorporate bone meal or fish meal into the soil.
        3. Ensure soil pH is between 6.0 and 7.0 for optimal phosphorus availability.
        """)
    elif P > 80:
        recommendations.append("""
        Phosphorus levels are high. Consider the following:
        1. Avoid phosphorus fertilizers temporarily.
        2. Plant phosphorus-hungry crops like potatoes or sunflowers.
        3. Consider using cover crops to prevent phosphorus runoff.
        """)
    
    # K recommendations
    if K < 100:
        recommendations.append("""
        Potassium levels are low. Consider the following:
        1. Apply potassium-rich fertilizers like potassium chloride or sulfate of potash.
        2. Incorporate wood ash or seaweed into the soil.
        3. Use organic mulches to slowly release potassium into the soil.
        """)
    elif K > 300:
        recommendations.append("""
        Potassium levels are high. Consider the following:
        1. Avoid potassium fertilizers temporarily.
        2. Ensure proper soil drainage to prevent potassium accumulation.
        3. Balance high potassium with calcium and magnesium applications.
        """)
    
    # pH recommendations
    if pH < 5.5:
        recommendations.append("""
        Soil pH is low (acidic). Consider the following:
        1. Apply agricultural lime to raise pH.
        2. Use dolomitic limestone if magnesium levels are also low.
        3. Avoid acidifying fertilizers like ammonium sulfate.
        """)
    elif pH > 7.5:
        recommendations.append("""
        Soil pH is high (alkaline). Consider the following:
        1. Apply sulfur or aluminum sulfate to lower pH.
        2. Use acidifying fertilizers like ammonium sulfate.
        3. Add organic matter to help buffer pH over time.
        """)
    
    # Crop-specific recommendations could be added here if 'crop' parameter is provided
    
    return "\n\n".join(recommendations)

# Example usage:
# recommendation = get_fertilizer_recommendation(N=45, P=75, K=200, pH=6.5, crop="tomato")
# print(recommendation)
