using LinearAlgebra
using Random
using Distributions

function noise(hometown)
    if hometown
        trunc_norm = Truncated(Normal(300, 20), 0, 1500)
        noise = rand(trunc_norm, 1)
    else
        trunc_norm = Truncated(Normal(300, 200), 0, 1500)
        noise = rand(trunc_norm, 1)
    end
    return noise
end

# reward assuming that you get a ticket
function reward(ticket, hometown, price, budget, days_till_concert)
    # no ticket, no reward
    if !ticket
        return 0
    end
    # you got the ticket
    i = 0
    if price > budget
        i = 1  # indicator variable that you paid less than your budget
    end
    if hometown
        reward = 1000 + (price - budget) * i + 0.5 * days_till_concert
    else
        reward = 800 + (price - budget) * i + 0.5 * days_till_concert
    end
    return reward
end


# using QLearning to model best action
function simple_model(budget, alpha)
    Q = zeros(268*2, 2)
    # this is the case that you got a Ticketmaster ticket on day 1
    ticket = rand(Bernoulli(0.01))
    for day=1:268
        s = 2 * day + 1 + ticket
        home_price = 100 + day + noise(1)
        away_price = 300 + day + noise(0)
        if ticket
            a = 1
            r = 0
            sp = s + 2
        else
            a = rand(Bernoulli(0.2)) + 1 # buy the ticket 20% of the time
            if a == 2
                sp = s + 3
            else
                sp = s + 2
            end
            price = min(home_price, away_price) # what price you are buying for

            # whether or not you bought in hometown
            if price == home_price
                hometown = 1
            else
                hometown = 0
            end
            r = reward(ticket, hometown, price, budget, 268 - day)
        end
        Q[s, a] += alpha * (r + maximum(Q[sp, :]) - Q[s, a])
    end
end