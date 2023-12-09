using LinearAlgebra
using Random
using Distributions
using StatsBase
using Plots
using DelimitedFiles
using CSV
using DataFrames

function cdf_correct(cdf_val)
    if cdf_val > 1
        cdf_val = 1.0
    end
    return round(cdf_val; digits=10)
end

"
Inputs: max_ticket_price, min_ticket_price, intervals (the ticket price interval you want you want)
Outputs: T_after_TM, T_first_fall, T_stable, T_rise, T_fall
    -- these are the transition matrices for the 4 periods of prices
        - after TM: the week or so directly following the ticketmaster sale craze
            - for T_{t, t+1}, modeled with truncated Normal with mean t + 40 and s.d. 20
        - first fall: the fall after the initial crazy for two weeks or so
            - for T_{t, t+1}, modeled with truncated Normal with mean t - 30 and s.d. 10
        - stable: the roughly stable ticket price
            - for T_{t, t+1}, modeled with truncated Normal with mean t and s.d. 50
        - rise: the rise in ticket prices the 3 or so weeks before the concert
            - for T_{t, t+1}, modeled with truncated Normal with mean t + 40 and s.d. 5 
        - fall: the fall in prices like 1/2 a week before the actual concert
            - for T_{t, t+1}, modeled with truncated Normal with mean t - 60 and s.d. 5 
"
function make_transition_matrices(max_ticket_price, min_ticket_price, intervals)
    size = convert(Int, 1 + (max_ticket_price - min_ticket_price) / intervals)
    # construct the transition matrices
    T_after_TM = zeros(BigFloat, size, size)
    T_first_fall = zeros(BigFloat, size, size)
    T_stable = zeros(BigFloat, size, size)
    T_rise = zeros(BigFloat, size, size)
    T_fall = zeros(BigFloat, size, size)
    half_int = intervals / 2

    for row=1:size
        row_price = min_ticket_price + intervals * (row - 1)

        # what the distributions we are drawing are
        after_TM_distr = Truncated(Normal(row_price + 50, 40), min_ticket_price - half_int, max_ticket_price + half_int)
        first_fall_distr =  Truncated(Normal(row_price - 30, 40), min_ticket_price - half_int, max_ticket_price + half_int)
        stable_distr = Truncated(Normal(row_price, 100), min_ticket_price - half_int, max_ticket_price + half_int)
        rise_distr = Truncated(Normal(row_price + 40, 40), min_ticket_price - half_int, max_ticket_price + half_int)
        fall_distr = Truncated(Normal(row_price - 200, 40), min_ticket_price - half_int, max_ticket_price + half_int)

        for col=1:size
            col_price = min_ticket_price + intervals * (col - 1)

            # taking CDF(price + 10) - CDF(price) to get PMF of the discrete area
            after_TM_first = cdf_correct(Distributions.cdf(after_TM_distr, col_price + half_int))
            after_TM_second = cdf_correct(Distributions.cdf(after_TM_distr, col_price - half_int))

            after_TM_discrete_distr = after_TM_first - after_TM_second
            T_after_TM[row, col] = after_TM_discrete_distr

            first_fall_first = cdf_correct(Distributions.cdf(first_fall_distr, col_price + half_int))
            first_fall_second = cdf_correct(Distributions.cdf(first_fall_distr, col_price - half_int))

            first_fall_discrete_distr = first_fall_first - first_fall_second
            T_first_fall[row, col] = first_fall_discrete_distr

            stable_first = cdf_correct(Distributions.cdf(stable_distr, col_price + half_int))
            stable_second = cdf_correct(Distributions.cdf(stable_distr, col_price - half_int))

            stable_discrete_distr =  stable_first - stable_second
            T_stable[row, col] = stable_discrete_distr

            rise_first = cdf_correct(Distributions.cdf(rise_distr, col_price + half_int))
            rise_second = cdf_correct(Distributions.cdf(rise_distr, col_price - half_int))

            rise_discrete_distr = rise_first - rise_second
            T_rise[row, col] = rise_discrete_distr

            fall_first = cdf_correct(Distributions.cdf(fall_distr, col_price + half_int))
            fall_second = cdf_correct(Distributions.cdf(fall_distr, col_price - half_int))
            
            fall_discrete_distr =  fall_first - fall_second
            T_fall[row, col] = fall_discrete_distr


        end

        # # normalize the row to a tolerance
        # normal_after_TM = normalize(T_after_TM[row, :], 1)
    end

    return T_after_TM, T_first_fall, T_stable, T_rise, T_fall
end


function price_simulation(max_ticket_price, min_ticket_price, price_interval, iteration, days)
    price_record = zeros(iteration, days)

    after_TM, first_fall, stable, rise, fall = make_transition_matrices(max_ticket_price, min_ticket_price, price_interval)

    possible_prices = size(after_TM, 1)

    # normalize the matrices, having problems creating them in helper function and moving over
    for matrix in [after_TM, first_fall, stable, rise, fall]
        for row=1:possible_prices
            for col=1:possible_prices
                if matrix[row, col] < 0
                    matrix[row, col] = 0
                end
            end
            matrix[row, :] = normalize(matrix[row, :], 1)
        end
    end

    for i=1:iteration

        # creating variation on when the different "periods" occur
        period_1_length = rand(DiscreteUniform(10, 30))
        period_2_length = rand(DiscreteUniform(10, 20))
        period_4_length = rand(DiscreteUniform(5, 15))
        period_5_length = rand(DiscreteUniform(2, 6))

        # ticket is indicator var of whether you have a ticket or not
        ticket = 0
        initial_ticket_price = Truncated(Normal(700, 80), 300, 1500)
        starting_price = rand(initial_ticket_price)
        price = round.(starting_price, digits=-1)
        # price = min_ticket_price + 5 * price_interval # need to change factor depending on size of price interval

        for j=1:days
            price_record[i, j] = price
            price_index = Int(round((price - min_ticket_price)/ price_interval)) + 1

            if j < period_1_length
                period = after_TM
            elseif j < period_1_length + period_2_length
                period = first_fall
            elseif j < days - period_4_length - period_5_length
                period = stable
            elseif j < days - period_5_length
                period = rise
            else
                period = fall
            end

            prob = period[price_index, :]
            row_dist = Categorical(prob) # sample from distribution above to find new ticket price

            new_price = (rand(row_dist) - 1) * price_interval + min_ticket_price
            price = new_price
        end
    end
    return(price_record)
end

"""Takes in the CSV file of the policy and gives you a matrix with rows = price_index and cols = day,
where value 1 corresponds to "A" and 0 corresponds to "B" """
function best_action_matrix(policy_csv, days, num_prices)

    matrix = CSV.read(policy_csv, DataFrame)
    matrix = Matrix(matrix)
    mapping_matrix = zeros(num_prices, days)
    for i=1:(size(matrix)[1] - 1)
        row = matrix[i, :]
        d = Int(row[1])
        p_val = row[2]
        p = Int((p_val - 100) / 10) + 1
        a = row[3]
        if a == "B"
            mapping_matrix[p, d] = 0
        else
            mapping_matrix[p, d] = 1
        end
    end
    return mapping_matrix
end

"""Creates heatmap of best action for each day/price. Red is buy and Blue is wait"""
function map_best_policy(policy_csv, days, num_prices)
    map = best_action_matrix(policy_csv, days, num_prices)
    custom_colormap = [RGB(1, 0, 0), RGB(0, 0, 1)]
    Plots.heatmap(map, c=custom_colormap, color=:auto, clear=true)
    title!("Best Action Per Day and Price")
    xlabel!("Days Since Ticket Release")
    ylabel!("Ticket Price (USD)")
    yticks!([1, 41, 91, 141, 191, 241, 291], ["100", "500", "1,000", "1,500", "2,000", "2,500", "3,000"])
    display(plot!())
end


"""Runs our policy against a naive policy of buying if under some budget. Output is the purchase price of policy. Returns inf price if you don't buy"""
function better_policy(iterations, policy_csv, days, num_prices, budget)

    purchase_price_naive = []
    purchase_price_policy = []

    action_matrix = best_action_matrix(policy_csv, days, num_prices)

    for k=1:iterations

        # indicator var of whether we have ticket yet
        naive_ticket = 0
        policy_ticket = 0

        price_record = price_simulation(3000, 100, 10, 1, 270)
        for d=1:days
            if naive_ticket == 1 && policy_ticket == 1 # in this case, you have ticket for both
                break
            end

            p = Int(price_record[1, d])
            relative_p = Int((p - 100) / 10) + 1

            # look at policy to see what action to take

            # if you don't have ticket, see if you should try to buy
            if naive_ticket == 0
                if p < budget
                    # only get ticket 95% of time if you take purchase action
                    get_ticket = 1 # rand(Bernoulli(.95))
                    if get_ticket == 1
                        naive_ticket = 1
                        push!(purchase_price_naive, p)
                    end
                end
            end

            # our policy
            if policy_ticket == 0
                a_policy = action_matrix[relative_p, d] # see what your best action is
                if a_policy == 1 # if you should buy, try that action
                    get_ticket = 1 # rand(Bernoulli(.95))
                    if get_ticket == 1 # if you get ticket, record the price
                        policy_ticket = 1
                        push!(purchase_price_policy, p)
                    end
                end
            end

            if d == days
                if naive_ticket == 0
                    push!(purchase_price_naive, inf)
                elseif policy_ticket == 0
                    push!(purchase_price_policy, inf)
                end
            end
        end

    end
    return purchase_price_naive, purchase_price_policy
end

purchase_price_naive, purchase_price_policy = better_policy(30, "/Users/Emily/Desktop/3ticket_buying_policy.csv", 270, 291, 800)
println(purchase_price_naive)
println(purchase_price_policy)