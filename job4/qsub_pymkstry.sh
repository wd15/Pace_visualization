# This is an example PBS script
#PBS -N thirtytwo
#PBS -l nodes=2:ppn=16
#PBS -l mem=96gb
#PBS -l walltime=24:00:00
#PBS -q hive-himem
#PBS -k oe
#PBS -m abe
#PBS -M beyucel@gatech.edu

cd $PBS_O_WORKDIR


module load anaconda3/2021.05

conda activate pymks

filename=pace-thirtytwo.csv

echo "chunks,samples,fit_time,predict_time" > ${filename}

for n_chunk in 1 2 4 8 16 24 32 40 48
do
    for repeat in {1..2}
    do
        for samples in 2000 4000 6000 8000
        do
            python main.py --samples=${samples} --chunks=${n_chunk} --filename=${filename} --threads_per_worker=2 --n_workers=32
	done
    done
done	
