import sys
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as la


def create_rating_matrix(user_artist_table):
    """
    converts the user_artist table to a rating matrix
    :param user_artist_table user_artist DataFrame
    :return: rating Dataframe
    """
    the_matrix = user_artist_table.pivot(index='userID', columns='artistID', values='weight')
    return the_matrix


def calc_average(rating_matrix):
    """
        calculate the average of all the ratings in the rating matrix
        :param rating_matrix matrix containing all the ratings
        :return: mean of the rating matrix
        """
    return np.nanmean(rating_matrix)


def calc_bibu_naive(r):
    """
        calculate users bias and artists bias by individual bias method
        :param r rating matrix
        :return: two dictionaries, one containing users biases and the other containing artists biases
        """
    bi_dict = {}
    bu_dict = {}
    r_avg = calc_average(r)
    for userID, user_artist in r.iterrows():
        bu_dict[userID] = np.nanmean(user_artist) - r_avg
    for artistID, user_artist in r.iteritems():
        bi_dict[artistID] = np.nanmean(user_artist) - r_avg
    return bi_dict, bu_dict


def calc_bi_bu_ls(rating_matrix, csv_file):
    """
        calculate users bias and artists bias by biases through least squares method
        :param rating_matrix rating matrix
        :csv_file user_artist DataFrame
        :return: two dictionaries, one containing users biases and the other containing artists biases
        """
    # csv_file_no_na = csv_file.dropna()
    print('starting to calc the bibus')
    r_avg = calc_average(rating_matrix)
    users_num = len(list(rating_matrix.index))
    columns = list(rating_matrix.index) + (list(rating_matrix.columns))

    # create A matrix
    A = pd.DataFrame(0, index=np.arange(len(csv_file)), columns=columns)
    for k in range(len(csv_file)):
        row = csv_file.iloc[k]
        A.at[k, row['userID']] = 1
        A.at[k, row['artistID']] = 1

    # create b vector
    b = rating_matrix.to_numpy().flatten(order='F')
    b = b[~np.isnan(b)]
    b -= r_avg

    print('starting least squares')

    # convert A to a sparse matrix
    A_converted = scipy.sparse.csr_matrix(A.values)

    x = la.lsqr(A_converted, b)[0]

    # devide the result of ls problem to users biases and artists biases
    bu_arr = x[:users_num]
    bi_arr = x[users_num:]
    bu_dict = {}
    bi_dict = {}
    for index, user in enumerate(rating_matrix.index):
        bu_dict[user] = bu_arr[index]
    for index, artist in enumerate(rating_matrix.columns):
        bi_dict[artist] = bi_arr[index]

    return bi_dict, bu_dict


def create_r_hat_df(rating_matrix, user_artist, bi_dict, bu_dict):
    """
    calculates prediction matrix
    :param rating_matrix: rating matrix
    :param user_artist: user_artist DataFrame
    :param bi_dict: dictionary containing artists biases
    :param bu_dict: dictionary containing users biases
    :return: prediction DataFrame
    """

    r_hat = rating_matrix.copy()
    r_avg = calc_average(rating_matrix)
    for k in range(len(user_artist)):
        u, i, w = user_artist.iloc[k]
        r_hat.at[u, i] = r_avg + bi_dict[i] + bu_dict[u]
    print('finished r_hat')
    return r_hat


def create_error_matrix(r, r_hat):
    """
       calculates error matrix
       :param r: rating matrix
       :param r_hat: prediction matrix
       :return: error DataFrame
       """
    return r.subtract(r_hat).fillna(0)


def create_similarity_matrix(r_tilda):
    """
       calculates similarity matrix
       :param r_tilda: error matrix
       :return: similarity matrix
       """
    print('starting sim_matrix')
    r_tilda_matrix = r_tilda.to_numpy()
    artist_count = r_tilda_matrix.shape[1]
    similarity_matrix = np.zeros((artist_count, artist_count))
    dot_products = np.matmul(r_tilda_matrix.T, r_tilda_matrix)
    couples = np.argwhere(dot_products != 0)
    for couple in couples:
        i, j = couple
        if similarity_matrix[i][j] != 0 or i == j:
            continue
        artists_matrix = np.array([r_tilda_matrix[:, i], r_tilda_matrix[:, j]]).T
        users_ranked_both_artists = artists_matrix[np.all(artists_matrix != 0, axis=1)]
        i_vec = users_ranked_both_artists[:, 0]
        j_vec = users_ranked_both_artists[:, 1]
        temp = dot_products[i][j] / (np.linalg.norm(i_vec) * np.linalg.norm(j_vec))
        if abs(temp) == 1:
            similarity_matrix[i][j] = 0
        else:
            similarity_matrix[i][j] = temp
        similarity_matrix[j][i] = similarity_matrix[i][j]
    print('finished sim matrix')
    return similarity_matrix


def create_index_dicts(user_artist):
    """
       creates dictionaries converting artistID to index and vice versa
       :param user_artist: user_artist DataFrame
       :return: two dictionaries
       """
    artist_to_index_dict = {}
    r = create_rating_matrix(user_artist)
    for index, artist in enumerate(r.columns):
        artist_to_index_dict[artist] = index
    index_to_artist_dict = {y: x for x, y in artist_to_index_dict.items()}
    return artist_to_index_dict, index_to_artist_dict


def calc_shuli(similarity_matrix, r_tilda, artist_to_index_dict, index_to_artist_dict, u, i, l_similar_artists):
    """
       calculates constant in prediction
       :param similarity_matrix: similarity matrix
       :param r_tilda: error matrix
       :param artist_to_index_dict: dictionary that converts artistID to index
       :param index_to_artist_dict: dictionary that converts index to artistID
       :param u: userID
       :param i: artistID
       :return: constant
       """
    shuli_lemala = 0
    shuli_lemata = 0
    for j in l_similar_artists:
        if j[0] != artist_to_index_dict[i]:
            shuli_lemala += similarity_matrix[artist_to_index_dict[i]][j[0]] * r_tilda.loc[u][
                index_to_artist_dict[j[0]]]
            shuli_lemata += abs(similarity_matrix[artist_to_index_dict[i]][j[0]])
    shuli = shuli_lemala / shuli_lemata
    return shuli


def calc_prediction(artist_to_index_dict, bi_dict, bu_dict, i, index_to_artist_dict, r, r_avg, r_hat, r_tilda,
                    similarity_matrix, u, l):
    """
      calculates prediction on user and artist couple
      :param artist_to_index_dict: dictionary that converts artistID to index
      :param bi_dict: dictionary containing artists biases
      :param bu_dict: dictionary containing users biases
      :param i: artistID
      :param r: raiting matrix
      :param r_avg: mean of rating matrix
      :param r_hat: prediction matrix
      :param r_tilda: error matrix
      :param similarity_matrix: similarity matrix
      :param u: userID
      :param l: number of neighbors
      """

    # predict by the following cases (as described in the pdf file)
    if i not in r.columns:
        if u not in r.index:
            r_hat.at[u, i] = r_avg
        else:
            r_hat.at[u, i] = r_avg + bu_dict[u]
    elif u not in r.index:
        r_hat.at[u, i] = r_avg + bi_dict[i]
    else:
        l_similar_artists = sorted(enumerate(similarity_matrix[artist_to_index_dict[i]]), key=lambda x: abs(x[1]),
                                   reverse=True)[:l]
        if max(l_similar_artists, key=lambda x: abs(x[1]))[1] == 0.0:
            r_hat.at[u, i] = r_avg + bi_dict[i] + bu_dict[u]
        else:
            r_hat.at[u, i] = r_avg + bi_dict[i] + bu_dict[u] + calc_shuli(similarity_matrix, r_tilda,
                                                                          artist_to_index_dict, index_to_artist_dict,
                                                                          u, i, l_similar_artists)


def log_ratings(user_artist):
    """
       calculates log on all ratings in the DataFrame
       :param user_artist: user artist DataFrame
       :return: user_artist DataFrame with log ratings
       """
    user_artist["weight"] = user_artist["weight"].apply(lambda x: 0.000000000001 if x == 1 else np.log10(x))
    return user_artist


def pow_ratings(test):
    """
       calculates power on all ratings in the DataFrame
       :param test: test DataFrame
       :return: test DataFrame with power ratings
       """
    test['predictions'] = test['predictions'].apply(lambda x: 10 ** x)


def train_test_split(user_artist):
    """
       split all data to train set and test set
        :param user_artist: user artist DataFrame
       :return: DataFrame containing train set, DtaFrame containing test set, and another containing the ratings
       of the test set
       """
    min_num_of_artist_listened = 35
    delete_num = 2
    validation_user = []
    validation_artist = []
    validation_weights = []
    train = user_artist.copy()

    for user in user_artist['userID'].drop_duplicates():
        artists_listened = user_artist.loc[user_artist["userID"] == user]["artistID"].tolist()
        if len(artists_listened) >= min_num_of_artist_listened:
            val_ratings = np.random.choice(
                artists_listened,
                size=delete_num,
                replace=False
            )
            to_validation = list(
                train.loc[(train["userID"] == user) & (train["artistID"].isin(list(val_ratings)))]['weight'])
            validation_weights.extend(to_validation)
            train.loc[(train["userID"] == user) & (train["artistID"].isin(list(val_ratings))), 'weight'] = np.nan
            validation_user.extend([user] * delete_num)
            validation_artist.extend(val_ratings)
    train.dropna(inplace=True)
    train.reset_index(inplace=True, drop=True)
    validation_df = pd.DataFrame(columns=["userID", "artistID"])
    validation_df["userID"] = validation_user
    validation_df["artistID"] = validation_artist
    return train, validation_df, validation_weights


def calc_rmse(val_weights, pred_weights):
    """
    calculates the RMSE rate
    :param val_weights: DataFrame containing true weights
    :param pred_weights: DataFrame containing predicted weights
    :return: RNSE rate
    """
    rmse = 0
    for val, pred in zip(val_weights, pred_weights):
        rmse += (val - pred) ** 2
    return (rmse / len(pred_weights)) ** 0.5


def calc_and_predict(artist_to_index_dict, bi_dict, bu_dict, index_to_artist_dict, r, r_avg, r_hat, r_tilda,
                     similarity_matrix, test):
    """
    calc predictions on test set
    :param artist_to_index_dict: dictionary that converts artistID to index
    :param bi_dict: dictionary containing artists biases
    :param bu_dict: dictionary containing users biases
    :param index_to_artist_dict: dictionary that converts index to artistID
    :param r: rating matrix
    :param r_avg: mean of rating matrix
    :param r_hat: prediction matrix
    :param r_tilda: error matrix
    :param similarity_matrix: similarity matrix
    :param test: DataFrame containing test set
    """
    for k in range(len(test)):
        u, i = test.iloc[k]

        #if the artist is new, add a new column to r_hat and r_tilda
        if i not in r_hat.columns:
            r_hat[i] = np.nan
            r_tilda[i] = np.nan
            index_to_artist_dict[len(artist_to_index_dict.keys())] = i
            artist_to_index_dict = {y: x for x, y in index_to_artist_dict.items()}

        #if the user is new, add a new column to r_hat and r_tilda
        if u not in r_hat.index:
            r_hat.at[u] = np.nan
            r_tilda.at[u] = np.nan

        calc_prediction(artist_to_index_dict, bi_dict, bu_dict, i, index_to_artist_dict, r, r_avg, r_hat, r_tilda,
                        similarity_matrix, u, 40)


def update_real_test(predictions, test_real):
    """
    :param predictions: DataFrame containing predictions
    :param test_real: test DataFrame
    :return: DataFrame containing all predictions
    """
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    test_real['weight'] = 0.0

    for k in range(len(test_real)):
        u, i, w = test_real.iloc[k]
        temp = predictions[(predictions.userID == u) & (predictions.artistID == i)]['predictions'].values[0]
        test_real.iloc[k, 2] = temp
    test_real['weight'] = test_real['weight'].apply(np.floor)

    # change all non-positive predictions to 1
    test_real['weight'] = test_real['weight'].mask(test_real['weight'].lt(0.1), 1)
    return test_real


def run_simulation(user_artist_train, test, test_weights, naive):
    """
    :param user_artist_train: user_artist DataFrame
    :param test: test DataFrame
    :param test_weights: real weights of the test set. if there is no test set, insert None
    :param naive: True if we are using the naive approach, False if we are using LS approach
    :return: test DataSet with predictions
    """
    print('starting run sim')
    log_ratings(user_artist_train)
    artist_to_index_dict, index_to_artist_dict = create_index_dicts(user_artist_train)
    r = create_rating_matrix(user_artist_train)

    if not naive:
        bi_dict, bu_dict = calc_bi_bu_ls(r, user_artist_train)
    else:
        bi_dict, bu_dict = calc_bibu_naive(r)

    r_avg = calc_average(r)
    r_hat = create_r_hat_df(r, user_artist_train, bi_dict, bu_dict)
    r_tilda = create_error_matrix(r, r_hat)
    similarity_matrix = create_similarity_matrix(r_tilda)
    calc_and_predict(artist_to_index_dict, bi_dict, bu_dict, index_to_artist_dict, r, r_avg, r_hat, r_tilda,
                     similarity_matrix, test)
    pred_list = []
    for k in range(len(test)):
        u, i = test.iloc[k]
        pred_list.append(r_hat[i][u])
    test['predictions'] = pred_list
    pow_ratings(test)

    if test_weights is not None:
        mse = calc_rmse(test_weights, test['predictions'])
        print('the mse is', mse)

    return test


if __name__ == '__main__':
    #load data
    user_artist = pd.read_csv('user_artist.csv')
    test = pd.read_csv('test.csv')
    test.drop_duplicates(inplace=True)

    #run simulation in test set and train set
    # train, validation_df, validation_weights = train_test_split(user_artist)
    # run_simulation(train, validation_df, validation_weights, True)


    our_test = run_simulation(user_artist, test, None, False)
    test_final = pd.read_csv('test.csv')

    result = update_real_test(our_test, test_final)

    #load prediction to csv
    result.to_csv(r'C:\test_hagasha.csv', index=False)
