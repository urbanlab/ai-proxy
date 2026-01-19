def calculate_token_cost(cost_per_million, token_nb):
    cost  = token_nb / 1000000 * cost_per_million
    return cost

    
