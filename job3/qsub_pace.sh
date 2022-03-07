# This is an example PBS script
#PBS -N twentyfournode
#PBS -l nodes=1:ppn=24
#PBS -l mem=96gb
#PBS -l walltime=15:00:00
#PBS -q hive
#PBS -k oe
#PBS -m abe
#PBS -M beyucel@gatech.edu

cd $PBS_O_WORKDIR


module load anaconda3/2020.11

conda env create -f environment.yml
conda activate pymks

filename=pace-twentyfournode.csv

echo "pca,chunks,samples,time" > ${filename}

conda activate pymks

for n_chunk in 1 2 4 8 16 24 32 40 48
do
    for repeat in {1..5}
    do
        for samples in 2000 4000 6000 8000
        do		       
            python pca.py --samples=${samples} --chunks=${n_chunk} --filename=${filename}
	done
    done
done	



