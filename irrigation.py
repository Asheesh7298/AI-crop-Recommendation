"""
Irrigation / water advice.

Estimates the crop's seasonal water need, subtracts what rainfall already
supplies, and turns the remainder into a simple irrigation schedule.
Optionally uses a soil-moisture reading (0-100%) to advise 'water now' or
'hold' - this is where an IoT soil-moisture sensor plugs in.
"""

from crops_meta import CROPS, WATER_MM


def advise(crop, rainfall_mm, soil_moisture=None):
    m = CROPS.get(crop, {})
    need_mm = WATER_MM.get(m.get("water", "medium"), 650)
    deficit = max(0, need_mm - rainfall_mm)

    tips = []
    tips.append(f"Seasonal water need for {crop}: ~{need_mm} mm ({m.get('water','medium')} demand).")
    if deficit <= 0:
        tips.append("Rainfall alone likely covers the requirement — minimal irrigation needed.")
    else:
        # ~50 mm per irrigation event as a rule of thumb
        events = max(1, round(deficit / 50))
        tips.append(f"Rainfall supplies ~{rainfall_mm:.0f} mm; irrigate ~{deficit:.0f} mm more "
                    f"across ~{events} watering(s) this season.")

    if soil_moisture is not None:
        if soil_moisture < 30:
            tips.append(f"💧 Soil moisture low ({soil_moisture:.0f}%) — water now.")
        elif soil_moisture > 70:
            tips.append(f"🚫 Soil moisture high ({soil_moisture:.0f}%) — hold irrigation, avoid waterlogging.")
        else:
            tips.append(f"👍 Soil moisture adequate ({soil_moisture:.0f}%).")
    return tips
