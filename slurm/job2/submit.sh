#!/bin/bash
#SBATCH --partition=rack6i               # -p, first available from the list of partitions
#SBATCH --time=04:00:00                 # -t, time (hh:mm:ss or dd-hh:mm:ss)
#SBATCH --nodes=1                       # -N, total number of machines
#SBATCH --ntasks=8  # -n, 64 MPI ranks per Opteron machine
#SBATCH --cpus-per-task=1               # threads per MPI rank
#SBATCH --job-name=pca                  # -J, for your records
#SBATCH --workdir=/working/wd15/git/Pace_visualization/slurm/job2  # -D, full path to an existing directory
#SBATCH --qos=test
#SBATCH --mem=0G

filename=slurm-${SLURM_JOB_ID}.csv

echo "chunks,samples,memory_usage,fit_time,predict_time,r_squared" > ${filename}

for n_chunk in 1 20 40 80 120 160 200 240 280 320
do
  for repeat in {1..2}
  do
    for samples in 2000 4000 6000 8000
    do
      mfilename=mprof.dat
      mprof run --output ${mfilename} --include-children ./main.py --samples=${samples} --chunks=${n_chunk} --filename=${filename} --threads_per_worker=2 --n_workers=16
      memory_value=`python memory.py --filename=mprof.dat`
      sed -i "s/memory_value/${memory_value}/" ${filename}
      \rm -rf mprof.dat
    done
  done
done

