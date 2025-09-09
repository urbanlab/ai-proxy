def calculate_co2_emission(watt, gram_per_kwh, duration):
    # return grams of CO2 emitted per kWh
    return (watt / 1000) * gram_per_kwh * duration
