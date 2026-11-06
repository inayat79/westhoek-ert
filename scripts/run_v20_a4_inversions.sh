#!/usr/bin/env bash
# v2.0 A4 (decision log v2.0-0): the missing λ of the outlier-treatment pools, same settings as the existing runs.
# Sequential (memory). New tags only: v20_<treatment>_lam<λ>.
cd "$(dirname "$0")"
LOG=../data/processed/v2_0/a4_inversions.log
mkdir -p ../data/processed/v2_0
COMMON="--para-depth 60 --para-dx 0.33 --err-rel 0.03 --max-iter 30"
run() {  # tag lam extra-args...
  tag=$1; lam=$2; shift 2
  if [ -f "../data/processed/runs/$tag/westhoek_inversion_results.npz" ]; then echo "$(date +%T) $tag exists" >> $LOG; return; fi
  echo "$(date +%T) start $tag" >> $LOG
  PYTHONIOENCODING=utf-8 MPLBACKEND=Agg OMP_NUM_THREADS=2 conda run -n pg python 02_land_inversion.py --tag $tag $COMMON --lam $lam "$@" > ../data/processed/v2_0/a4_$tag.out 2>&1
  echo "$(date +%T) done $tag exit $?" >> $LOG
}
for L in 5 10 20 100; do run v20_g1b_lam$L $L --drop-list ../results/bel_v3/g1b_droplist_thr299.txt; done
for L in 5 10 50 100; do run v20_g2_lam$L $L --drop-list ../results/deep_all_G2_droplist_flagged_electrodes.txt; done
for L in 5 10 50 100; do run v20_g3_lam$L $L --err-scale-file ../results/deep_all_G3_err_scale.txt; done
for L in 5 10 20 100; do run v20_recip_lam$L $L --drop-list ../results/deep_all_droplist.txt --err-scale-file ../data/processed/v1_4/recip_err_scale.txt; done
echo "$(date +%T) all done" >> $LOG
