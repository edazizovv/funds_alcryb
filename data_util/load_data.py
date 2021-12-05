#


#
import pandas


#
from merrill_feature.feature_treatment.treat import standard_treat, lag, wise_drop, add_by_substring


#
def cry_load(bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/dataset_groupgrab.csv'
    data = pandas.read_csv(g)
    data = data.set_index("Timestamp").sort_index()

    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    data = data.dropna()

    """
    Feature Treatment
    """

    data['USD'] = 1

    no, diff, b100diff, b100, no__, diff__, b100diff__, b100__ = [], [], [], [], [], [], [], []

    cols = data.columns.values
    no = add_by_substring(cols, no__, no)
    diff = add_by_substring(cols, diff__, diff)
    b100diff = add_by_substring(cols, b100diff__, b100diff)
    b100 = add_by_substring(cols, b100__, b100)
    pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

    data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)
    data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
    data_pct_lagged = data_pct_lagged.dropna()

    excluded = []

    # stats
    wise = []

    x_factors = [x for x in data_pct_lagged.columns.values if 'LAG' in x and ~any(
        [y in x for y in excluded + wise])]

    X = data_pct_lagged.loc[:, x_factors].values
    Y = data_pct_lagged.loc[:, [target0, target1]].values
    X_ = data_pct_lagged.loc[:, [target0, target1]].values
    Y_ = data_pct_lagged.loc[:, [target0, target1]].values

    tt = data_pct_lagged.index.values

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged, x_factors

