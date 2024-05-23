#!/bin/bash

# generate training data
#for ((i=30 ; i<=33 ; i++))
#do
#./predictor ../../CBP-16-Simulation-master/cbp2016.eval/traces/SHORT_MOBILE-"${i}".bt9.trace.gz SHORT_MOBILE-"${i}".bt9.trace.gz
#done

for i in 1 13 28 43 60 51 34
do
./predictor ../../CBP-16-Simulation-master/cbp2016.eval/traces/SHORT_MOBILE-"${i}".bt9.trace.gz SHORT_MOBILE-"${i}".bt9.trace.gz
done


# generate testing data
#for ((i=10 ; i<=10 ; i++))
#do
#./predictor ../../CBP-16-Simulation-master/cbp2016.eval/evaluationTraces/SHORT_MOBILE-"${i}".bt9.trace.gz SHORT_MOBILE-"${i}".bt9.trace.gz 1
#done

for i in 15 30 39 43
do
./predictor ../../CBP-16-Simulation-master/cbp2016.eval/evaluationTraces/SHORT_MOBILE-"${i}".bt9.trace.gz SHORT_MOBILE-"${i}".bt9.trace.gz 1
done