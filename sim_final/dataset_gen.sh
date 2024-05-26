#!/bin/bash

# this might cause out of memory problem
preprocess_data_1() {
    local input_file="$1"
    local output_file="$2"
    local threshold="$3"

    echo "preprocess data input : $input_file, output : $output_file"

    awk -v threshold="$threshold" '{count[$0]++} END {for (line in count) print count[line], line}' "$input_file" | sort -k2 | awk -v threshold="$threshold" '{
    if ($1 > threshold) {
        count=threshold
    } else {
        count=$1
    }
    for (i=1; i<=count; i++) print $2
    }' > $output_file
}

preprocess_data() {
    local input_file="$1"
    local output_file="$2"
    local threshold="$3"

    echo "preprocess data input : $input_file, output : $output_file"
    sort "$input_file" | uniq -c > "$input_file-tmp"
    echo "sorting data done, extracting data from $input_file-tmp"

    awk -v threshold="$threshold" '{
        if ($1 > threshold) {
            count=threshold
        } else {
            count=$1
        }
        for (i=1; i<=count; i++) print $2
    }' "$input_file-tmp" > $output_file
}

# generating training data function
generate_training_data() {
    

    local prefix="$1"
    local limit="$2"
    local inner_limit=$((limit/10))

    echo "===${prefix}===" >> "../data/train/data"

    for((i=0;i<=inner_limit;i++)); do
        # preprocess 10 traces at a time
        for((j=i*10+1;j<=(i+1)*10;j++)); do
            if ((j<=limit)); then
                ./predictor ../../CBP-16-Simulation-master/cbp2016.eval/traces/"${prefix}"-"${j}".bt9.trace.gz "${prefix}"-"${j}".bt9.trace.gz
                wc -l ../data/train/"${prefix}"-"${j}".bt9.trace.gz.csv >> "../data/train/data"
                preprocess_data ../data/train/"${prefix}"-"${j}".bt9.trace.gz.csv ../data/train/"${prefix}"-"${j}".bt9.trace 8
                wc -l ../data/train/"${prefix}"-"${j}".bt9.trace >> "../data/train/data"
                rm ../data/train/"${prefix}"-"${j}".bt9.trace.gz.csv
                #cat ../data/train/"${prefix}"-"${j}".bt9.trace >> ../data/train/processed/combined-"${prefix}"-"${i}"
                #rm ../data/train/"${prefix}"-"${j}".bt9.trace
            fi
        done

        for((j=i*10+1;j<=(i+1)*10;j++)); do
            if ((j<=limit)); then
                cat ../data/train/"${prefix}"-"${j}".bt9.trace >> ../data/train/processed/combined-"${prefix}"-"${i}"
                rm ../data/train/"${prefix}"-"${j}".bt9.trace
            fi
        done

        preprocess_data ../data/train/processed/combined-"${prefix}"-"${i}" ../data/train/processed/"${prefix}"-"${i}" 60
        wc -l ../data/train/processed/"${prefix}"-"${i}" >> "../data/train/data"
        shuf ../data/train/processed/"${prefix}"-"${i}" > ../data/train/processed/"${prefix}"-shuf"${i}"
        rm ../data/train/processed/combined-"${prefix}"-"${i}"
        rm ../data/train/processed/"${prefix}"-"${i}"

        #wc -l ../data/train/processed/combined-"${prefix}"-"${i}" >> "../data/train/data"
        #shuf ../data/train/processed/combined-"${prefix}"-"${i}" > ../data/train/processed/"${prefix}"-shuf"${i}"
        #rm ../data/train/processed/combined-"${prefix}"-"${i}"

    done

    for ((i=0;i<=inner_limit;i++))
    do
        cat ../data/train/processed/"${prefix}"-shuf"${i}" >> ../data/train/processed/"${prefix}"-shuf-combined
        rm ../data/train/processed/"${prefix}"-shuf"${i}"
    done

    preprocess_data ../data/train/processed/"${prefix}"-shuf-combined ../data/train/processed/"${prefix}"-shuf 100
    wc -l ../data/train/processed/"${prefix}"-shuf >> "../data/train/data"
    shuf ../data/train/processed/"${prefix}"-shuf > ../data/train/processed/"${prefix}"-fin
    #rm ../data/train/processed/"${prefix}"-shuf-combined
    split --numeric-suffixes --filter='gzip > $FILE.gz' -a 5 -l 50000000 ../data/train/processed/"${prefix}"-fin ../data/train/processed/"${prefix}"-split
    #rm ../data/train/processed/"${prefix}"-fin

}

# generating training data function
generate_testing_data() {

    local prefix="$1"
    local limit="$2"

    for((i=1;i<=limit;i++)); do 

        local input_file="../data/test/"${prefix}"-"${i}".bt9.trace.gz.csv"
        local tmp_file="../data/test/"${prefix}"-"${i}".bt9.trace.tmp"
        local output_file="../data/test/"${prefix}"-"${i}".bt9.trace"

        ./predictor ../../CBP-16-Simulation-master/cbp2016.eval/evaluationTraces/"${prefix}"-"${i}".bt9.trace.gz "${prefix}"-"${i}".bt9.trace.gz 1
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

generate_testing_data "SHORT_MOBILE" 10
#generate_testing_data "SHORT_SERVER" 293
#generate_testing_data "LONG_MOBILE" 32
#generate_testing_data "LONG_SERVER" 8
