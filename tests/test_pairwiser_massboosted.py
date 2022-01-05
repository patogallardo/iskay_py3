import numpy as np
import iskay.pairwiser_massboosted as pairwiser_massboosted
import iskay.pairwiser as pairwiser
import numba


def test_pairwiser_one_row_uneven_bins():
    length = 10000
    row = np.random.randint(0, length)
    Dc = np.random.uniform(low=100, high=110, size=length)
    mass = np.ones(length)
    ra_rad = np.random.uniform(low=0, high=2*np.pi, size=length)
    dec_rad = np.random.uniform(low=-30, high=30, size=length)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    nrbin = 40
    binsz = 5
    bin_edges = np.arange(0, binsz*(nrbin+1.0), binsz)
    dTw_pairwise_one_row = np.zeros(nrbin)
    w2_pairwise_one_row = np.zeros(nrbin)
    # mass boosted output
    dTw_pariwise_one_row_massb = np.zeros(nrbin)
    w2_pairwise_one_row_massb = np.zeros(nrbin)
    npairs = np.zeros(nrbin)
    sum_mass = np.zeros(nrbin)
    pairwiser.pairwise_one_row(row, Dc, ra_rad, dec_rad, tzav,
                               Tmapsc, nrbin, binsz, dTw_pairwise_one_row,
                               w2_pairwise_one_row)
    pairwiser_massboosted.pairwiser_massboost_one_row(row, Dc, ra_rad,
                                                      dec_rad, tzav,
                                                      Tmapsc, bin_edges,
                                                      mass,
                                           dTw_pariwise_one_row_massb,  # noqa
                                           w2_pairwise_one_row_massb,  # noqa
                                           sum_mass,  # noqa
                                           npairs)  # noqa
    sum_err_sq1 = np.sum((dTw_pairwise_one_row - # noqa
                          dTw_pariwise_one_row_massb)**2)
    sum_err_sq2 = np.sum((w2_pairwise_one_row -  # noqa
                          w2_pairwise_one_row_massb)**2)
    assert sum_err_sq1 < 1e-10 and sum_err_sq2 < 1e-10


def test_pairwiser_mass_boosted():
    length = 10000
    Dc = np.random.uniform(low=100, high=110, size=length)
    ra_deg = np.random.uniform(low=0, high=359, size=length)
    dec_deg = np.random.uniform(low=-30, high=30, size=length)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    mass = np.ones(length)

    nrbin = 40
    binsz = 5
    bin_edges = np.arange(0, binsz*(nrbin + 1.0), binsz)

    rsep, pest_m = pairwiser_massboosted.pairwise_ksz_massboosted(Dc, ra_deg,
                 dec_deg, tzav, Tmapsc, mass, bin_edges,  # noqa
                 Nthreads=numba.config.NUMBA_NUM_THREADS)  # noqa
    rsep, pest = pairwiser.pairwise_ksz(Dc, ra_deg, dec_deg, tzav,
                                        Tmapsc, binsz, nrbin,
                                        multithreading=False)
    diff_sq = (pest - pest_m)**2
    assert np.sum(diff_sq) < 1e-10
