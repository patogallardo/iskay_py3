qsub  -b y -cwd -N run00 /nfs/user/pag227/miniconda/bin/python ./iskay_analysis.py params_disjoint_bin_lum_gt_04p3_and_06p1_bs_dt.ini
qsub  -b y -cwd -N run01 -hold_jid run00 /nfs/user/pag227/miniconda/bin/python ./iskay_analysis.py params_disjoint_bin_lum_gt_06p1_and_07p9_bs_dt.ini
qsub  -b y -cwd -N run02 -hold_jid run01 /nfs/user/pag227/miniconda/bin/python ./iskay_analysis.py params_lum_gt_04p3_bs_dt.ini
qsub  -b y -cwd -N run03 -hold_jid run02 /nfs/user/pag227/miniconda/bin/python ./iskay_analysis.py params_lum_gt_06p1_bs_dt.ini
qsub  -b y -cwd -N run04 -hold_jid run03 /nfs/user/pag227/miniconda/bin/python ./iskay_analysis.py params_lum_gt_07p9_bs_dt.ini