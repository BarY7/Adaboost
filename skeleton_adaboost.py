#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    S_len = len(X_train)
    Dt = np.array([1/S_len for i in range(S_len)])
    hypo_array = []
    alpha_vals = []
    for i in range(T):
        print("##### ADABOOST RUN ITERATION t={} ######".format(i))
        ht, ht_score = compute_wl(X_train, y_train, 5000, Dt)
        et = 1 - ht_score
        wt = (1/2) * np.log((1-et)/et)
        Dt_next_helper = np.array([compute_next_Dt_for_sample(
            Dt, wt, k, X_train[k], y_train[k], ht) for k in range(S_len)])
        Zt = Dt_next_helper.sum()
        Dt = Dt_next_helper/Zt
        hypo_array.append(translate_ht(ht))
        alpha_vals.append(wt)
    return (hypo_array, alpha_vals)


##############################################
# You can add more methods here, if needed.

def reverse_ht(ht):
    lt_flag = True if ht[0] == 1 else False
    return (ht[1], ht[2], lt_flag)


def translate_ht(ht):
    lt_flag = ht[2]
    return (1, ht[0], ht[1]) if lt_flag else (-1, ht[0], ht[1])


def compute_next_Dt_for_sample(Dt, wt, xi_index, xi, yi, hypo):
    yi_guess = compute_hypo_output(xi, hypo[0], hypo[1], hypo[2])
    return Dt[xi_index] * np.exp(-wt * yi * yi_guess)


def compare_by_lt_flag(x_test_coor, theta, lt_flag):
    return x_test_coor <= theta if lt_flag else x_test_coor >= theta


def compute_hypo_output(x, word_index, theta, lt_flag):
    x_test_coor = x[word_index]
    y_guess = compare_by_lt_flag(x_test_coor, theta, lt_flag)
    return 1 if y_guess else -1


def compute_reversed_hypo_output(x, hypo):
    ht = reverse_ht(hypo)
    return compute_hypo_output(x, ht[0], ht[1], ht[2])


def compute_hypo_weighted_score(x_train, y_train, Dt, word_index, theta, lt_flag):
    sum_right = 0
    for i in range(len(x_train)):
        x, y = x_train[i], y_train[i]
        y_guess = compute_hypo_output(
            x, word_index, theta=theta, lt_flag=lt_flag)
        if(y == y_guess):
            sum_right = sum_right + Dt[i]
    weighted_accuracy = sum_right
    return weighted_accuracy


def compute_wl(x_train, y_train, vocabulary_len, Dt):
    best_hypo = (0, 0, True)  # word_index, theta, lt_flag
    best_hypo_score = 0
    for word_index in range(vocabulary_len):
        for i in range(int(np.max(x_train, axis=0)[word_index])):
            theta = i
            cur_hypo_score = compute_hypo_weighted_score(
                x_train, y_train, Dt, word_index, theta, True)
            if(cur_hypo_score >= best_hypo_score):
                best_hypo_score = cur_hypo_score
                best_hypo = (word_index, theta, True)
            cur_hypo_score = compute_hypo_weighted_score(
                x_train, y_train, Dt, word_index, theta, False)
            if(cur_hypo_score >= best_hypo_score):
                best_hypo_score = cur_hypo_score
                best_hypo = (word_index, theta, False)
    return best_hypo, best_hypo_score


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    # a
    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # t_array = range(T)
    # t_train_err_arr = []
    # t_test_err_arr = []
    # for t in t_array:
    #     print("##### COMPUTE ERROR FOR t={} ######".format(t),)
    #     t_train_err = compute_t_time_err(
    #         X_train, y_train, t, hypotheses, alpha_vals)
    #     t_test_err = compute_t_time_err(
    #         X_test, y_test, t, hypotheses, alpha_vals)
    #     t_train_err_arr.append(t_train_err)
    #     t_test_err_arr.append(t_test_err)
    # plt.ylabel('train_err')
    # plt.xlabel('t')
    # plt.plot(t_array, t_train_err_arr)
    # plt.show()
    # plt.ylabel('test_err')
    # plt.xlabel('t')
    # plt.plot(t_array, t_test_err_arr)
    # plt.show()

    #b
    
    # T = 10
    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # for i in range(len(hypotheses)):
    #     print("***** Word ******")
    #     print(vocab[hypotheses[i][1]])
    #     print(hypotheses[i][2])
    #     print(hypotheses[i][0])
    #     print("------------------------------------------")


    # #c
    T=80
    m = len(X_train) 
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    t_array = range(T)
    l_array = []
    for t in range(0,T):
        sum = 0
        for i in range(m):
            sum = sum + c_compute_exp(t,X_train[i],y_train[i],hypotheses,alpha_vals)
        l_array.append(sum/m)
    plt.ylabel('train loss')
    plt.xlabel('t')
    plt.plot(t_array, l_array)
    plt.show();

    m = len(X_test)
    l_array = []
    for t in range(0,T):
        sum = 0
        for i in range(m):
            sum = sum + c_compute_exp(t,X_test[i],y_test[i],hypotheses,alpha_vals)
        l_array.append(sum/m)
    plt.ylabel('test loss')
    plt.xlabel('t')
    plt.plot(t_array, l_array)
    plt.show()
    

        

    ##############################################
    # You can add more methods here, if needed.


def compute_t_time_err(x_arr, y_arr, t, hypotheses, alpha_vals):
    sum_right = 0
    for i in range(len(x_arr)):
        x, y = x_arr[i], y_arr[i]
        guess_raw_val = np.array(
            [alpha_vals[k] * compute_reversed_hypo_output(x, hypotheses[k]) for k in range(t+1)])
        y_guess = 1 if np.sum(guess_raw_val) >= 0 else -1
        sum_right = sum_right + 1 if (y == y_guess) else sum_right
    return sum_right/len(x_arr)

def c_compute_exp(t, xi, yi ,hypotheses, alpha_vals):
    sum = 0
    for j in range(t):
        y_guess = compute_reversed_hypo_output(xi, hypotheses[j])
        sum = sum + alpha_vals[j] * y_guess * -yi
    return np.exp(sum)


    ##############################################
if __name__ == '__main__':
    main()
