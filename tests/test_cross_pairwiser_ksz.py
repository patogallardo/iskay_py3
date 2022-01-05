import numpy as np
import iskay.pairwiser as pairwiser
import iskay.cross_ksz_pairwiser as cross_ksz_pairwiser
from iskay import catalogTools
import os
from iskay import paramTools


def test_pairwiser_one_row_uneven_bins():
    length = 10000
    row = np.random.randint(0, length)
    Dc = np.random.uniform(low=100, high=110, size=length)
    ra_rad = np.random.uniform(low=0, high=2*np.pi, size=length)
    dec_rad = np.random.uniform(low=-30, high=30, size=length)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    nrbin = 40
    binsz = 5
    bin_edges = np.arange(0, binsz*(nrbin+1.0), binsz)

    dTw_pariwise_one_row_unevenBins = np.zeros(nrbin)
    w2_pairwise_one_row_unevenBins = np.zeros(nrbin)
    dTw_pariwise_one_row_unevenBins_cross = np.zeros(nrbin)
    w2_pairwise_one_row_unevenBins_cross = np.zeros(nrbin)

    pairwiser.pairwise_one_row_uneven_bins(row, Dc, ra_rad, dec_rad, tzav,
                                           Tmapsc, bin_edges,
                                           dTw_pariwise_one_row_unevenBins,
                                           w2_pairwise_one_row_unevenBins)
    cross_ksz_pairwiser.cross_pairwiser_one_row_uneven_bins(row, Dc,
                        ra_rad, dec_rad, tzav, tzav, Tmapsc, Tmapsc,  # noqa
                        bin_edges, dTw_pariwise_one_row_unevenBins_cross,  # noqa
                        w2_pairwise_one_row_unevenBins_cross)  # noqa

    sum_err_sq1 = np.sum((dTw_pariwise_one_row_unevenBins_cross - # noqa
                          dTw_pariwise_one_row_unevenBins)**2)
    sum_err_sq2 = np.sum((w2_pairwise_one_row_unevenBins_cross -  # noqa
                          w2_pairwise_one_row_unevenBins)**2)

    assert sum_err_sq2 < 1e-10
    assert sum_err_sq1 < 1e-10


def produceFakeCatalog():
    ''' Returns a fake pandas dataframe with data for pairwiser_ksz'''
    #produce fake data
    from iskay import cosmology
    import pandas as pd

    Nobj = 10000
    z = np.random.uniform(0, 1, Nobj)
    Dc = cosmology.Dc(z)
    ra_deg = np.random.uniform(0, 350, Nobj)
    dec_deg = np.random.uniform(-30, 0, Nobj)
    dT = np.random.uniform(-300, 300, Nobj)
    datain = {'z': z, 'Dc': Dc, 'ra': ra_deg, 'dec': dec_deg, 'dT': dT}
    df = pd.DataFrame(datain)
    return df
    #end produce fake data


def test_get_cross_pairwise_ksz_even_weights():
    testPath = '/'.join((catalogTools.__file__).split('/')[:-2]) + '/tests/'
    testParamFileFullPath = os.path.join(testPath,
                                         'data_toTestAPI/params.ini')
    params = paramTools.params(testParamFileFullPath)

    df = produceFakeCatalog()

    tzav = pairwiser.get_tzav(df.dT.values, df.z.values, params.SIGMA_Z)

    rsep0, p_uk0 = pairwiser.pairwise_ksz_uneven_bins(df.Dc.values,
                                                      df.ra.values,
                                                      df.dec.values,
                                                      tzav,
                                                      df.dT.values,
                                                      params.BIN_EDGES,
                                                      do_variance_weighted=False,  # noqa
                                                      divs=None,
                                                      multithreading=False)

    rsep, p_uk = cross_ksz_pairwiser.get_cross_pairwise_ksz(df, df, params)
    rsep_diff_sq = np.sum((rsep - rsep0)**2)
    p_uk_diff_sq = np.sum((p_uk - p_uk0)**2)
    assert rsep_diff_sq < 1e-10
    assert p_uk_diff_sq < 1e-10
