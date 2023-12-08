import pandas as pd 
import numpy as np 
import csv

MIN_PRICE = 100
MAX_PRICE = 3000
PRICE_INCREMENT = 10
NUM_PRICE_PTS = 290     # discretized possible ticket prices
NUM_DAYS = 270

NUM_ACTIONS = 2    # buy (b) or wait (w)
NUM_STATES = NUM_DAYS * (MAX_PRICE - MIN_PRICE) // PRICE_INCREMENT + 1  # + 1 for absorbing state 

EPSILON = 0.99
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.1 

NUM_ITERATIONS = 500

Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# def get_state(p, d): 
#     s = (d - 1)*(NUM_PRICE_PTS) + ((p - MIN_PRICE) / (PRICE_INTERVAL))   ## ?? CONFIRM WITH EMILY 

def encode_state(d, p):
    if d < NUM_DAYS: 
        return (d * (MAX_PRICE - MIN_PRICE) // PRICE_INCREMENT + (p - MIN_PRICE) // PRICE_INCREMENT)
    else: 
        return NUM_STATES - 1   # Absorbing state -- last state 

def decode_state(s):
    if s < NUM_STATES - 1:  ## Not absorbing state
        d = s // (MAX_PRICE - MIN_PRICE) // PRICE_INCREMENT
        p = (s % ((MAX_PRICE - MIN_PRICE) // PRICE_INCREMENT)) * PRICE_INCREMENT + MIN_PRICE
        return d, p 
    else:   ## Absorbing state
        return "Absorbing state"

def get_reward(s, a):
    if s == NUM_STATES - 1:     ## Absorbing state
        return 0    # No reward in absorbing state 
    d, p = decode_state(s)
    if a == 0: # Buy
        #return MAX_PRICE - p    # Higher reward for lower price 
        return (MAX_PRICE - p) // (PRICE_INCREMENT)
    else:   # Wait
        if d == NUM_DAYS - 1: 
            return -MAX_PRICE   # Large penalty for not buying a ticket by the last day
        else: 
            return -1 
            #return (-10 - d) // (PRICE_INCREMENT)      # Small penalty for waiting, increases as concert date approaches

def QLearning(df):
    for i in range(NUM_ITERATIONS): 
        prices = df.iloc[[i]]   ## df of prices over 270 days from row i (therefore iteration i)
        #print(prices)
        state = int(encode_state(0, prices.iloc[0, 0]))  ## start state: day 0 with ticket price 0  
        #print("START PRICE: ", prices.iloc[0, 0])
        #print("START STATE: ", state)

        while state != NUM_STATES - 1:  # Continue until absorbing state
            current_day, current_price = decode_state(state)
            #print("CURRENT DAY: ", current_day, "CURRENT PRICE: ", current_price)

            ## Epsilon greedy strategy for action selection: 
            if np.random.rand() < EPSILON: 
                action = np.random.choice(NUM_ACTIONS)    # Explore: Random action
            else: 
                action = np.argmax(Q[state, :])   # Exploit best action based on Q-value

            reward = get_reward(state, action)
            if action == 0 or current_day == NUM_DAYS - 1:  # If ticket is bought or it's the last day 
                next_state = NUM_STATES - 1     # Move to absorbing state
            else: 
                next_price = prices.iloc[0, current_day + 1]      ## extract from df 
                next_state = encode_state(current_day + 1, next_price)

            # if state < 0 or state >= NUM_STATES or next_state < 0 or next_state >= NUM_STATES:
            #     raise ValueError(f"State index out of bounds: state={state}, next_state={next_state}")

            ## Q-Learning update: 
            Q[state, action] = (1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE *(reward + DISCOUNT_FACTOR * np.max(Q[int(next_state), :]))

            state = int(next_state) 

    policy = [np.argmax(Q[s, :]) for s in range(NUM_STATES - 1)]
    #decoded_policy = [(decode_state(s), "B" if a == 0 else "W") for s, a in enumerate(policy)]
    decoded_policy = [] 
    for s in range(NUM_STATES - 1): 
        d, p = decode_state(s)
        a = policy[s]
        a_str = "B" if a == 0 else "W"
        decoded_policy.append((d, p, a_str))

    csv_file_path = '3ticket_buying_policy.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the header
        writer.writerow(['Day', 'Price', 'Action'])

        # Write the policy data
        for day, price, action in decoded_policy:
            writer.writerow([day, price, action])

    print(f"Policy saved to {csv_file_path}")

        
def get_data(fp): 
    df = pd.read_csv(fp)
    return df 

    #print(df)
    names = list(df.columns)
    row_0 = df.iloc[[0]]
    #print(row_0)
    row_0.to_csv("row_0.csv", index=False)

# function to make decisions 
# q learning data format should be: s, a, r, s, p


def main():
    fp = "./data/ticket_prices_2.csv"
    #fp = "/Users/Jessica/Documents/CS238/AA228-CS238-Student-master/finalProject/data/ticket_price_1.csv"
    df = get_data(fp)
    QLearning(df)

if __name__ == '__main__':
    main() 