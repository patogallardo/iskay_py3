from iskay import mapTools

map_fname = ('actpol/map_coadd/20200228/'
             'release/act_planck_s08_s18_cmb_f150_night_map.fits')


#def test_openMap_local():
#    theMap = mapTools.openMap_local(map_fname)
#    assert len(theMap.shape) == 2


def test_openMap_remote():
    theMap = mapTools.openMap_remote(map_fname)
    assert len(theMap.shape) == 2
