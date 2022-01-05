import pandas as pd
import numpy as np
from iskay import tiled_JK


def fakeDataset():
    '''Generates a fake dataset with ra, and dec to test the pipeline'''
    ra = np.random.uniform(0, 10, 1000)
    dec = np.random.uniform(0, 10, 1000)
    df = pd.DataFrame({'ra': ra, 'dec': dec})
    return df


def test_classify_grid():
    '''
    On a fake dataset, classify the grid specifying the Nside of the grid.
    '''
    df = fakeDataset()
    l1 = df.shape[0]

    df = tiled_JK.classify_grid(df, Nside=16)
    l2 = df.shape[0]

    assert l2 == l1
    assert 'JK_index' in df.columns


def test_histogram_catalog():
    '''Make a histogram per healpix grid of the current dataset.'''
    df = fakeDataset()
    Nside = 32
    m = tiled_JK.healpix_histogram_catalog(df, Nside=Nside)
    assert len(m) > 0


def test_remove_edge_galaxies():
    '''Removes the edge galaxies in the 2d histogram, by cutting within
    a sigma tolerance of the typicial number of objects in a bin.'''
    df = fakeDataset()
    Nside = 32
    df1 = tiled_JK.remove_edge_galaxies(df, tol_sigma=2, Nside=Nside)
    df2 = tiled_JK.remove_edge_galaxies(df, tol_sigma=0.3, Nside=Nside)
    assert len(df1) >= len(df2)


def test_how_many_tiles():
    '''HOw many tiles in a tiledjk object.'''
    df = fakeDataset()
    df = tiled_JK.classify_grid(df, 16)
    Ntiles = tiled_JK.how_many_tiles(df)
    assert Ntiles > 0


def test_remove_tile():
    '''Tests method to remove one tile'''
    df = fakeDataset()
    df = tiled_JK.classify_grid(df, Nside=16)
    to_remove = 0
    df_new = tiled_JK.remove_tile(df, to_remove)
    assert len(df_new) < len(df)


def test_getSide_given_iterations():
    '''Test the iteration solver.'''
    df = fakeDataset()
    Npix = 300
    Nside = tiled_JK.getSide_given_iterations(df, Npix)
    assert Nside > 0
