#!/bin/bash

preprocess_data() {
    local input_file="$1"
    local output_file="$2"
    local threshold="$3"

    echo "preprocess data input : $input_file, output : $output_file"
    sort "$input_file" | uniq -c > "$input_file-tmp"
    echo "sorting data done, extracting data from $input_file-tmp"
    rm "$input_file"

    awk -v threshold="$threshold" '{
        if ($1 > threshold) {
            count=threshold
        } else {
            count=$1
        }
        for (i=1; i<=count; i++) print $2
    }' "$input_file-tmp" > $output_file
    rm "$input_file-tmp"
}

# generating training data function
generate_training_data() {
    

    local prefix="$1"
    local limit="$2"
    local inner_limit=$((limit/10))

    echo "===${prefix}===" >> "../data/"${prefix}".train/data"

    for((i=0;i<=inner_limit;i++)); do
        # preprocess 10 traces at a time
        for((j=i*10+1;j<=(i+1)*10;j++)); do
            if ((j<=limit)); then
                ./predictor ../data/traces/"${prefix}"-"${j}".bt9.trace.gz "${prefix}"-"${j}".bt9.trace.gz "${prefix}"
                wc -l ../data/"${prefix}".train/"${prefix}"-"${j}".bt9.trace.gz.csv >> "../data/"${prefix}".train/data"
                preprocess_data ../data/"${prefix}".train/"${prefix}"-"${j}".bt9.trace.gz.csv ../data/"${prefix}".train/"${prefix}"-"${j}".bt9.trace 100
                wc -l ../data/"${prefix}".train/"${prefix}"-"${j}".bt9.trace >> "../data/"${prefix}".train/data"
            fi
        done

        for((j=i*10+1;j<=(i+1)*10;j++)); do
            if ((j<=limit)); then
                cat ../data/"${prefix}".train/"${prefix}"-"${j}".bt9.trace >> ../data/"${prefix}".train/processed/combined-"${prefix}"-"${i}"
                rm ../data/"${prefix}".train/"${prefix}"-"${j}".bt9.trace
            fi
        done

        preprocess_data ../data/"${prefix}".train/processed/combined-"${prefix}"-"${i}" ../data/"${prefix}".train/processed/"${prefix}"-"${i}" 400
        wc -l ../data/"${prefix}".train/processed/"${prefix}"-"${i}" >> "../data/"${prefix}".train/data"
        shuf ../data/"${prefix}".train/processed/"${prefix}"-"${i}" > ../data/"${prefix}".train/processed/"${prefix}"-shuf-"${i}"
        #rm ../data/"${prefix}".train/processed/combined-"${prefix}"-"${i}"
        rm ../data/"${prefix}".train/processed/"${prefix}"-"${i}"

    done

    for ((i=0;i<=inner_limit;i++))
    do
        cat ../data/"${prefix}".train/processed/"${prefix}"-shuf-"${i}" >> ../data/"${prefix}".train/processed/"${prefix}"-shuf-combined
        rm ../data/"${prefix}".train/processed/"${prefix}"-shuf-"${i}"
    done

    # since it might encounter OOM-kill during shuffle the dataset
    # we can remove the data after generating is done
    preprocess_data ../data/"${prefix}".train/processed/"${prefix}"-shuf-combined ../data/"${prefix}".train/processed/"${prefix}"-shuf 600
    wc -l ../data/"${prefix}".train/processed/"${prefix}"-shuf >> "../data/"${prefix}".train/data"
    shuf ../data/"${prefix}".train/processed/"${prefix}"-shuf > ../data/"${prefix}".train/processed/"${prefix}"-fin
    #rm ../data/"${prefix}".train/processed/"${prefix}"-shuf-combined
    split --numeric-suffixes -l 1000000 ../data/"${prefix}".train/processed/"${prefix}"-fin ../data/"${prefix}".train/processed/"${prefix}"-fin-
    #rm ../data/"${prefix}".train/processed/"${prefix}"-fin

}

# generating training data function
generate_testing_data() {

    local prefix="$1"
    local limit="$2"

    for((i=14;i<=limit;i++)); do 

        local input_file="../data/"${prefix}".test/"${prefix}"-"${i}".bt9.trace.gz.csv"
        local tmp_file="../data/"${prefix}".test/"${prefix}"-"${i}".bt9.trace.tmp"
        local output_file="../data/"${prefix}".test/"${prefix}"-"${i}".bt9.trace"

        ./predictor ../data/evaluationTraces/"${prefix}"-"${i}".bt9.trace.gz "${prefix}"-"${i}".bt9.trace.gz "${prefix}" 1
        sort $input_file | uniq -c > $tmp_file
        awk '{ print $1","$2 }' $tmp_file > $output_file

        if [ -s "$input_file" ] ; then rm $input_file ; fi
        if [ -s "$tmp_file" ] ; then rm $tmp_file ; fi

    done
}

#generate_training_data "SHORT_MOBILE" 61
#generate_training_data "SHORT_SERVER" 139
#generate_training_data "LONG_MOBILE" 19
#generate_training_data "LONG_SERVER" 4

#generate_testing_data "SHORT_MOBILE" 107
#generate_testing_data "SHORT_SERVER" 293
#generate_testing_data "LONG_MOBILE" 32
#generate_testing_data "LONG_SERVER" 8
