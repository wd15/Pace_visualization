filename=pace-sixteennode.csv

echo "chunks,samples,fit_time,predict_time" > ${filename}

for n_chunk in 1 2 4 8 16 24 32 40 48
do
    for repeat in {1..2}
    do
        for samples in 2000 4000 6000 
        do		       
            python main.py --samples=${samples} --chunks=${n_chunk} --filename=${filename} --threads_per_worker=2 --n_workers=16
	done
    done
done	


