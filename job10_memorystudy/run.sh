filename=output.csv

echo "chunks,samples,memory_usage,fit_time,predict_time" > ${filename}

samples=2000
xdatafile='x_data.zarr'
ydatafile='y_data.npy'
for n_chunk in 1 2 4 8 16 25 40
do
    for repeat in {1..5}
    do
        \rm -rf ${xdatafile}
        \rm -rf ${ydatafile}
        python prepare.py --samples=${samples} --chunks=${n_chunk} --threads_per_worker=1 --n_workers=2
        mfilename=mprof.dat
        mprof run --output ${mfilename} --include-children ./main.py --samples=${samples} --chunks=${n_chunk} --filename=${filename} --threads_per_worker=1 --n_workers=2
        memory_value=`python memory.py --filename=mprof.dat`
        sed -i "s/memory_value/${memory_value}/" ${filename}
        \rm -rf mprof.dat
    done
done
