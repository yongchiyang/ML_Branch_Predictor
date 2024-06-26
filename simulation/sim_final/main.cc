///////////////////////////////////////////////////////////////////////
//  Copyright 2015 Samsung Austin Semiconductor, LLC.                //
///////////////////////////////////////////////////////////////////////

//Description : Main file for CBP2016 

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <iostream>
#include <fstream>
using namespace std;

#include "utils.h"
//#include "bt9.h"
#include "bt9_reader.h"
//#include "predictor.cc"
#include "predictor.h"


#define COUNTER     unsigned long long
#define WRITE_CSV


void CheckHeartBeat(UINT64 numIter, UINT64 numMispred)
{
  UINT64 dotInterval=1000000;
  UINT64 lineInterval=30*dotInterval;

 UINT64 d1K   =1000;
 UINT64 d10K  =10000;
 UINT64 d100K =100000;
 UINT64 d1M   =1000000; 
 UINT64 d10M  =10000000;
 UINT64 d30M  =30000000;
 UINT64 d60M  =60000000;
 UINT64 d100M =100000000;
 UINT64 d300M =300000000;
 UINT64 d600M =600000000;
 UINT64 d1B   =1000000000;
 UINT64 d10B  =10000000000;


//  if(numIter % lineInterval == 0){ //prints line every 30 million branches
//    printf("\n");
//    fflush(stdout);
//  }
  if(numIter == d1K){ //prints MPKI after 100K branches
    printf("  MPKBr_1K         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }

  if(numIter == d10K){ //prints MPKI after 100K branches
    printf("  MPKBr_10K         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }
  
  if(numIter == d100K){ //prints MPKI after 100K branches
    printf("  MPKBr_100K         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }
  if(numIter == d1M){
    printf("  MPKBr_1M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter)); 
    fflush(stdout);
  }

  if(numIter == d10M){ //prints MPKI after 100K branches
    printf("  MPKBr_10M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }

  if(numIter == d30M){ //prints MPKI after 100K branches
    printf("  MPKBr_30M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }

  if(numIter == d60M){ //prints MPKI after 100K branches
    printf("  MPKBr_60M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }

  if(numIter == d100M){ //prints MPKI after 100K branches
    printf("  MPKBr_100M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }
  
  if(numIter == d300M){ //prints MPKI after 100K branches
    printf("  MPKBr_300M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }

  if(numIter == d600M){ //prints MPKI after 100K branches
    printf("  MPKBr_600M         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }

  if(numIter == d1B){ //prints MPKI after 100K branches
    printf("  MPKBr_1B         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }
  
  if(numIter == d10B){ //prints MPKI after 100K branches
    printf("  MPKBr_10B         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(numIter));   
    fflush(stdout);
  }
 
}//void CheckHeartBeat

// usage: predictor <trace>

int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("usage: %s <trace>\n", argv[0]);
    exit(-1);
  }
  
  ///////////////////////////////////////////////
  // Init variables
  ///////////////////////////////////////////////
    
    PREDICTOR  *brpred = new PREDICTOR();  // this instantiates the predictor code
  ///////////////////////////////////////////////
  // read each trace recrod, simulate until done
  ///////////////////////////////////////////////

    std::string trace_path;
    trace_path = argv[1];

#ifdef WRITE_CSV
    std::string trace_name;
    std::string trace_type;
    std::string fileName;
    if(argc >= 4){
      trace_name = argv[2]; 
      trace_type = argv[3];
      if (argc == 5)
        fileName = "../data/" + trace_type + ".test/" + trace_name + ".csv";
      else
        fileName = "../data/" + trace_type + ".train/" + trace_name + ".csv";
    }
    else fileName = "../data/test.csv";
#endif

    bt9::BT9Reader bt9_reader(trace_path);

    std::string key = "total_instruction_count:";
    std::string value;
    bt9_reader.header.getFieldValueStr(key, value);
    UINT64     total_instruction_counter = std::stoull(value, nullptr, 0);
    UINT64 current_instruction_counter = 0;
    key = "branch_instruction_count:";
    bt9_reader.header.getFieldValueStr(key, value);
    UINT64     branch_instruction_counter = std::stoull(value, nullptr, 0);
    UINT64     numMispred =0;  
//ver2    UINT64     numMispred_btbMISS =0;  
//ver2    UINT64     numMispred_btbANSF =0;  
//ver2    UINT64     numMispred_btbATSF =0;  
//ver2    UINT64     numMispred_btbDYN =0;  

    UINT64 cond_branch_instruction_counter=0;
//ver2     UINT64 btb_ansf_cond_branch_instruction_counter=0;
//ver2     UINT64 btb_atsf_cond_branch_instruction_counter=0;
//ver2     UINT64 btb_dyn_cond_branch_instruction_counter=0;
//ver2     UINT64 btb_miss_cond_branch_instruction_counter=0;
           UINT64 uncond_branch_instruction_counter=0;

//ver2    ///////////////////////////////////////////////
//ver2    // model simple branch marking structure
//ver2    ///////////////////////////////////////////////
//ver2    std::map<UINT64, UINT32> myBtb; 
//ver2    map<UINT64, UINT32>::iterator myBtbIterator;
//ver2
//ver2    myBtb.clear();
   
  ///////////////////////////////////////////////
  // read each trace record, simulate until done
  ///////////////////////////////////////////////

      OpType opType;
      UINT64 PC;
      bool branchTaken;
      UINT64 branchTarget;
      UINT64 numIter = 0;

#ifdef WRITE_CSV
      ofstream fileObj;
      fileObj.open(fileName);
#endif
   
      for (auto it = bt9_reader.begin(); it != bt9_reader.end(); ++it) {
        CheckHeartBeat(++numIter, numMispred); //Here numIter will be equal to number of branches read

        try {
          bt9::BrClass br_class = it->getSrcNode()->brClass();

//          bool dirDynamic = (it->getSrcNode()->brObservedTakenCnt() > 0) && (it->getSrcNode()->brObservedNotTakenCnt() > 0); //JD2_2_2016
//          bool dirNeverTkn = (it->getSrcNode()->brObservedTakenCnt() == 0) && (it->getSrcNode()->brObservedNotTakenCnt() > 0); //JD2_2_2016

//JD2_2_2016 break down branch instructions into all possible types
          opType = OPTYPE_ERROR; 

          if ((br_class.type == bt9::BrClass::Type::UNKNOWN) && (it->getSrcNode()->brNodeIndex())) { //only fault if it isn't the first node in the graph (fake branch)
            opType = OPTYPE_ERROR; //sanity check
          }
//NOTE unconditional could be part of an IT block that is resolved not-taken
//          else if (dirNeverTkn && (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)) {
//            opType = OPTYPE_ERROR; //sanity check
//          }
//JD_2_22 There is a bug in the instruction decoder used to generate the traces
//          else if (dirDynamic && (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)) {
//            opType = OPTYPE_ERROR; //sanity check
//          }
          else if (br_class.type == bt9::BrClass::Type::RET) {
            if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
              opType = OPTYPE_RET_COND;
            else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
              opType = OPTYPE_RET_UNCOND;
            else {
              opType = OPTYPE_ERROR;
            }
          }
          else if (br_class.directness == bt9::BrClass::Directness::INDIRECT) {
            if (br_class.type == bt9::BrClass::Type::CALL) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
                opType = OPTYPE_CALL_INDIRECT_COND;
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
                opType = OPTYPE_CALL_INDIRECT_UNCOND;
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else if (br_class.type == bt9::BrClass::Type::JMP) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
                opType = OPTYPE_JMP_INDIRECT_COND;
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
                opType = OPTYPE_JMP_INDIRECT_UNCOND;
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else {
              opType = OPTYPE_ERROR;
            }
          }
          else if (br_class.directness == bt9::BrClass::Directness::DIRECT) {
            if (br_class.type == bt9::BrClass::Type::CALL) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) {
                opType = OPTYPE_CALL_DIRECT_COND;
              }
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) {
                opType = OPTYPE_CALL_DIRECT_UNCOND;
              }
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else if (br_class.type == bt9::BrClass::Type::JMP) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) {
                opType = OPTYPE_JMP_DIRECT_COND;
              }
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) {
                opType = OPTYPE_JMP_DIRECT_UNCOND;
              }
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else {
              opType = OPTYPE_ERROR;
            }
          }
          else {
            opType = OPTYPE_ERROR;
          }

  
          PC = it->getSrcNode()->brVirtualAddr();

          branchTaken = it->getEdge()->isTakenPath();
          branchTarget = it->getEdge()->brVirtualTarget();

          //printf("PC: %llx type: %x T %d N %d outcome: %d", PC, (UINT32)opType, it->getSrcNode()->brObservedTakenCnt(), it->getSrcNode()->brObservedNotTakenCnt(), branchTaken);

/************************************************************************************************************/

          if (opType == OPTYPE_ERROR) { 
            if (it->getSrcNode()->brNodeIndex()) { //only fault if it isn't the first node in the graph (fake branch)
              fprintf(stderr, "OPTYPE_ERROR\n");
              printf("OPTYPE_ERROR\n");
              exit(-1); //this should never happen, if it does please email CBP org chair.
            }
          }
          else if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) { //JD2_17_2016 call UpdatePredictor() for all branches that decode as conditional
            //printf("COND ");

//NOTE: contestants are NOT allowed to use the btb* information from ver2 of the infrastructure below:
//ver2             myBtbIterator = myBtb.find(PC); //check BTB for a hit
//ver2            bool btbATSF = false;
//ver2            bool btbANSF = false;
//ver2            bool btbDYN = false;
//ver2
//ver2            if (myBtbIterator == myBtb.end()) { //miss -> we have no history for the branch in the marking structure
//ver2              //printf("BTB miss ");
//ver2              myBtb.insert(pair<UINT64, UINT32>(PC, (UINT32)branchTaken)); //on a miss insert with outcome (N->btbANSF, T->btbATSF)
//ver2              predDir = brpred->GetPrediction(PC, btbANSF, btbATSF, btbDYN);
//ver2              brpred->UpdatePredictor(PC, opType, branchTaken, predDir, branchTarget, btbANSF, btbATSF, btbDYN); 
//ver2            }
//ver2            else {
//ver2              btbANSF = (myBtbIterator->second == 0);
//ver2              btbATSF = (myBtbIterator->second == 1);
//ver2              btbDYN = (myBtbIterator->second == 2);
//ver2              //printf("BTB hit ANSF: %d ATSF: %d DYN: %d ", btbANSF, btbATSF, btbDYN);
//ver2
//ver2              predDir = brpred->GetPrediction(PC, btbANSF, btbATSF, btbDYN);
//ver2              brpred->UpdatePredictor(PC, opType, branchTaken, predDir, branchTarget, btbANSF, btbATSF, btbDYN); 
//ver2
//ver2              if (  (btbANSF && branchTaken)   // only exhibited N until now and we just got a T -> upgrade to dynamic conditional
//ver2                 || (btbATSF && !branchTaken)  // only exhibited T until now and we just got a N -> upgrade to dynamic conditional
//ver2                 ) {
//ver2                myBtbIterator->second = 2; //2-> dynamic conditional (has exhibited both taken and not-taken in the past)
//ver2              }
//ver2            }
//ver2            //puts("");

            bool predDir = false;
#ifdef WRITE_CSV
            //fileObj << (PC>>2) % (1 << 8) << ",";
            brpred->Write_Data(PC,fileObj);
            if(branchTaken) fileObj << "1\n";
            else fileObj << "0\n";
#endif
            predDir = brpred->GetPrediction(PC);
            brpred->UpdatePredictor(PC, opType, branchTaken, predDir, branchTarget); 

            if(predDir != branchTaken){
              numMispred++; // update mispred stats
//ver2              if(btbATSF)
//ver2                numMispred_btbATSF++; // update mispred stats
//ver2              else if(btbANSF)
//ver2                numMispred_btbANSF++; // update mispred stats
//ver2              else if(btbDYN)
//ver2                numMispred_btbDYN++; // update mispred stats
//ver2              else
//ver2                numMispred_btbMISS++; // update mispred stats
            }
            cond_branch_instruction_counter++;

//ver2            if (btbDYN)
//ver2              btb_dyn_cond_branch_instruction_counter++; //number of branches that have been N at least once after being T at least once
//ver2            else if (btbATSF)
//ver2              btb_atsf_cond_branch_instruction_counter++; //number of branches that have been T at least once, but have not yet seen a N after the first T
//ver2            else if (btbANSF)
//ver2              btb_ansf_cond_branch_instruction_counter++; //number of cond branches that have not yet been observed T
//ver2            else
//ver2              btb_miss_cond_branch_instruction_counter++; //number of cond branches that have not yet been observed T
          }
          else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) { // for predictors that want to track unconditional branches
            uncond_branch_instruction_counter++;
            brpred->TrackOtherInst(PC, opType, branchTaken, branchTarget);
          }
          else {
            fprintf(stderr, "CONDITIONALITY ERROR\n");
            printf("CONDITIONALITY ERROR\n");
            exit(-1); //this should never happen, if it does please email CBP org chair.
          }

/************************************************************************************************************/
        }
        catch (const std::out_of_range & ex) {
          std::cout << ex.what() << '\n';
          break;
        }
      
      } //for (auto it = bt9_reader.begin(); it != bt9_reader.end(); ++it)
#ifdef WRITE_CSV
      fileObj.close();
#endif

    ///////////////////////////////////////////
    //print_stats
    ///////////////////////////////////////////

    //NOTE: competitors are judged solely on MISPRED_PER_1K_INST. The additional stats are just for tuning your predictors.

      printf("  TRACE \t : %s" , trace_path.c_str()); 
      printf("  NUM_INSTRUCTIONS            \t : %10llu",   total_instruction_counter);
      printf("  NUM_BR                      \t : %10llu",   branch_instruction_counter-1); //JD2_2_2016 NOTE there is a dummy branch at the beginning of the trace...
      printf("  NUM_UNCOND_BR               \t : %10llu",   uncond_branch_instruction_counter);
      printf("  NUM_CONDITIONAL_BR          \t : %10llu",   cond_branch_instruction_counter);
//ver2      printf("  NUM_CONDITIONAL_BR_BTB_MISS \t : %10llu",   btb_miss_cond_branch_instruction_counter);
//ver2      printf("  NUM_CONDITIONAL_BR_BTB_ANSF \t : %10llu",   btb_ansf_cond_branch_instruction_counter);
//ver2      printf("  NUM_CONDITIONAL_BR_BTB_ATSF \t : %10llu",   btb_atsf_cond_branch_instruction_counter);
//ver2      printf("  NUM_CONDITIONAL_BR_BTB_DYN  \t : %10llu",   btb_dyn_cond_branch_instruction_counter);
      printf("  NUM_MISPREDICTIONS          \t : %10llu",   numMispred);
//ver2      printf("  NUM_MISPREDICTIONS_BTB_MISS \t : %10llu",   numMispred_btbMISS);
//ver2      printf("  NUM_MISPREDICTIONS_BTB_ANSF \t : %10llu",   numMispred_btbANSF);
//ver2      printf("  NUM_MISPREDICTIONS_BTB_ATSF \t : %10llu",   numMispred_btbATSF);
//ver2      printf("  NUM_MISPREDICTIONS_BTB_DYN  \t : %10llu",   numMispred_btbDYN);
      printf("  MISPRED_PER_1K_INST         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(total_instruction_counter));
//ver2      printf("  MISPRED_PER_1K_INST_BTB_MISS\t : %10.4f",   1000.0*(double)(numMispred_btbMISS)/(double)(total_instruction_counter));
//ver2      printf("  MISPRED_PER_1K_INST_BTB_ANSF\t : %10.4f",   1000.0*(double)(numMispred_btbANSF)/(double)(total_instruction_counter));
//ver2      printf("  MISPRED_PER_1K_INST_BTB_ATSF\t : %10.4f",   1000.0*(double)(numMispred_btbATSF)/(double)(total_instruction_counter));
//ver2      printf("  MISPRED_PER_1K_INST_BTB_DYN \t : %10.4f",   1000.0*(double)(numMispred_btbDYN)/(double)(total_instruction_counter));
      printf("\n");
}



