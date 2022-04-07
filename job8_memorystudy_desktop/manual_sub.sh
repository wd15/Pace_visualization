filename=OUTPUT.csv


echo "chunks,samples,memory_usage,fit_time,predict_time,r_squared" > ${filename}

for n_chunk in 1 2 4 16 32
do
    for repeat in {1..2}
    do
        for samples in 2000 4000 8000
        do
            mfilename=mprof.dat
            mprof run --output ${mfilename} --include-children ./main.py --samples=${samples} --chunks=${n_chunk} --filename=${filename} --threads_per_worker=2 --n_workers=4
            memory_value=`python memory.py --filename=mprof.dat`
            sed -i "s/memory_value/${memory_value}/" ${filename}
            \rm -rf mprof.dat
	done
    done
done
