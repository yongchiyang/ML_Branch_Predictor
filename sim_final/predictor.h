///////////////////////////////////////////////////////////////////////
////  Copyright 2015 Samsung Austin Semiconductor, LLC.                //
/////////////////////////////////////////////////////////////////////////
//
#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "utils.h"
#include "bt9.h"
#include "bt9_reader.h"

#define PHT_CTR_MAX  3
#define PHT_CTR_INIT 2

#define HIST_LEN   17

//NOTE competitors are allowed to change anything in this file include the following two defines
//ver2 #define FILTER_UPDATES_USING_BTB     0     //if 1 then the BTB updates are filtered for a given branch if its marker is not btbDYN
//ver2 #define FILTER_PREDICTIONS_USING_BTB 0     //if 1 then static predictions are made (btbANSF=1 -> not-taken, btbATSF=1->taken, btbDYN->use conditional predictor)

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

class PREDICTOR{

  // The state is defined for Gshare, change for your design

 private:
  UINT32  ghr;           // global history register
  UINT32  *pht;          // pattern history table
  UINT32  *lpht;         // local pattern history table
  UINT32  *bimodal;
  UINT32  historyLength; // history length
  UINT32  numPhtEntries; // entries in pht 
  UINT16  *ga;
  UINT32  *lht;
  INT32   gaLength;

 public:


  PREDICTOR(void);
  //NOTE contestants are NOT allowed to use these versions of the functions
//ver2   bool    GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN);  
//ver2   void    UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN);
//ver2   void    TrackOtherInst(UINT64 PC, OpType opType, bool branchDir, UINT64 branchTarget);

  // The interface to the functions below CAN NOT be changed
  bool    GetPrediction(UINT64 PC);  
  void    UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget);
  void    TrackOtherInst(UINT64 PC, OpType opType, bool branchDir, UINT64 branchTarget);
  void    HistoryUpdate(UINT64 PC, OpType opType, bool resolveDir, UINT64 branchTarget);
  void    Write_Data(UINT64 PC,ofstream& fileObj);
  // Contestants can define their own functions below
};

/////////////// STORAGE BUDGET JUSTIFICATION ////////////////
// Total storage budget: 32KB + 17 bits
// Total PHT counters: 2^17 
// Total PHT size = 2^17 * 2 bits/counter = 2^18 bits = 32KB
// GHR size: 17 bits
// Total Size = PHT size + GHR size
/////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

PREDICTOR::PREDICTOR(void){

  historyLength    = HIST_LEN;
  ghr              = 0;
  numPhtEntries    = (1<< HIST_LEN);
  gaLength         = 8;

  pht = new UINT32[numPhtEntries];
  ga = new UINT16[gaLength];
  lht = new UINT32[numPhtEntries];
  lpht = new UINT32[numPhtEntries];
  bimodal = new UINT32[numPhtEntries];

  for(UINT32 ii=0; ii< numPhtEntries; ii++){
    pht[ii]=PHT_CTR_INIT; 
    lpht[ii]=PHT_CTR_INIT;
    bimodal[ii]=PHT_CTR_INIT;
    lht[ii]=0;
  }
  
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

//ver2 version of infrastructure. Contestants are NOT allowed to use this version of the function
//ver2 bool   PREDICTOR::GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN){
//ver2 
//ver2   UINT32 phtIndex   = (PC^ghr) % (numPhtEntries);
//ver2   UINT32 phtCounter = pht[phtIndex];
//ver2 
//ver2 //  printf(" ghr: %x index: %x counter: %d prediction: %d\n", ghr, phtIndex, phtCounter, phtCounter > PHT_CTR_MAX/2);
//ver2 
//ver2   //NOTE taking advantage of btbANFS and btbATSF to make static predictions based upon previously observed behaviour. You don't have to take advantage of this if you don't want to.
//ver2   if(btbANSF && FILTER_PREDICTIONS_USING_BTB) {
//ver2     return NOT_TAKEN; //statically predict N if it has always been N since discovery if BTB prediction filtering is enabled
//ver2   }
//ver2   else if(btbATSF && FILTER_PREDICTIONS_USING_BTB) {
//ver2     return TAKEN; //statically predict T if it has always been T since discovery if BTB prediction filtering is enabled
//ver2   }
//ver2   else if (btbDYN || !FILTER_PREDICTIONS_USING_BTB) { //use PHT if it is marked as dynamic in the btb
//ver2     if(phtCounter > (PHT_CTR_MAX/2)){ 
//ver2       return TAKEN; 
//ver2     }
//ver2     else{
//ver2       return NOT_TAKEN; 
//ver2     }
//ver2   }
//ver2 
//ver2   //statically predict N if is not marked at all in the btb
//ver2   return NOT_TAKEN; 
//ver2 }

//NOTE contestants are not allowed to use the btb* information from ver2 of the simulation infrastructure. Interface to this function CAN NOT be changed.
bool   PREDICTOR::GetPrediction(UINT64 PC){

  UINT32 phtIndex   = ((PC>>2)^ghr) % (numPhtEntries);
  UINT32 lphtIndex  = lht[phtIndex] % (numPhtEntries);
  UINT32 phtCounter = pht[phtIndex];
  UINT32 lphtCounter = lpht[lphtIndex];
  UINT32 biCounter = bimodal[phtIndex];

//  printf(" ghr: %x index: %x counter: %d prediction: %d\n", ghr, phtIndex, phtCounter, phtCounter > PHT_CTR_MAX/2);
  if(bimodal[phtIndex] & 0b10){
    if(lphtCounter  & 0b10){ 
      return TAKEN; 
    }
    else{
      return NOT_TAKEN; 
    }
  }
  else{
    if(phtCounter  & 0b10){ 
      return TAKEN; 
    }
    else{
      return NOT_TAKEN; 
    }
  }
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

//ver2 version of infrastructure. Contestants are NOT allowed to use this version of the function
//ver2 void  PREDICTOR::UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN){
//ver2 
//ver2   UINT32 phtIndex   = (PC^ghr) % (numPhtEntries);
//ver2   UINT32 phtCounter = pht[phtIndex];
//ver2 
//ver2   //NOTE only updating PHT and GHR if branch has exhibited dynamic behaviour in the past OR has always been taken until the current not-taken. This is optional, just showing what you could do with this info
//ver2 
//ver2   if(btbDYN || (resolveDir != predDir) || !FILTER_UPDATES_USING_BTB) { 
//ver2   // update the PHT if btb indicates dynamic behaviour OR the outcome does not match the prediction
//ver2     if(resolveDir == TAKEN){
//ver2       pht[phtIndex] = SatIncrement(phtCounter, PHT_CTR_MAX);
//ver2     }else{
//ver2       pht[phtIndex] = SatDecrement(phtCounter);
//ver2     }
//ver2 
//ver2   // update the GHR
//ver2     ghr = (ghr << 1);
//ver2 
//ver2     if(resolveDir == TAKEN){
//ver2       ghr++; 
//ver2     }
//ver2   }
//ver2 
//ver2 }

//NOTE contestants are not allowed to use the btb* information from ver2 of the simulation infrastructure. Interface to this function CAN NOT be changed.
void  PREDICTOR::UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget){

  UINT32 phtIndex   = ((PC>>2)^ghr) % (numPhtEntries);
  UINT32 lphtIndex  = lht[phtIndex] % (numPhtEntries);
  UINT32 phtCounter = pht[phtIndex];
  UINT32 lphtCounter = lpht[lphtIndex];
  UINT32 biCounter = bimodal[phtIndex];

  bool lphtPred = (lphtCounter > (PHT_CTR_MAX/2));
  bool phtPred = (phtCounter > (PHT_CTR_MAX/2));
  if(lphtPred != phtPred){
    if(resolveDir == lphtPred){
      bimodal[phtIndex] = SatIncrement(biCounter, PHT_CTR_MAX);
    }
    else{
      bimodal[phtIndex] = SatDecrement(biCounter);
    }
  }

  if(resolveDir == TAKEN){
    pht[phtIndex] = SatIncrement(phtCounter, PHT_CTR_MAX);
    lpht[lphtIndex] = SatIncrement(lphtCounter, PHT_CTR_MAX);
  }else{
    pht[phtIndex] = SatDecrement(phtCounter);
    lpht[lphtIndex] = SatDecrement(lphtCounter);
  }

  HistoryUpdate(PC,opType,resolveDir,branchTarget);
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void    PREDICTOR::TrackOtherInst(UINT64 PC, OpType opType, bool branchDir, UINT64 branchTarget){

  // This function is called for instructions which are not
  // conditional branches, just in case someone decides to design
  // a predictor that uses information from such instructions.
  // We expect most contestants to leave this function untouched.
  HistoryUpdate(PC,opType,branchDir,branchTarget);
  return;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void    PREDICTOR::HistoryUpdate(UINT64 PC, OpType opType, bool resolveDir, UINT64 branchTarget){

  UINT32 phtIndex   = ((PC>>2)^ghr) % (numPhtEntries);
  // update the GHR
  ghr = (ghr << 1);
  // update local history table
  lht[phtIndex] = (lht[phtIndex] << 1);

  if(resolveDir == TAKEN){
    ghr++; 
    lht[phtIndex] ++;
  }


  //for(int i = 0; i < gaLength-1; i++){
  //  ga[i] = ga[i+1];
  //}

  //ga[gaLength-1] = (PC>>2) & 0b11111111;
  return;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void    PREDICTOR::Write_Data(UINT64 PC,ofstream& fileObj){

  UINT32 phtIndex   = ((PC>>2)^ghr) % (numPhtEntries);
  UINT32 lphtIndex  = lht[phtIndex] % (numPhtEntries);
  UINT32 phtCounter = pht[phtIndex];
  UINT32 lphtCounter = lpht[lphtIndex];

  //fileObj << (ghr & 0b11111111) << ",";
  // global history
  for(int i = HIST_LEN-1; i >= 0; --i){
    if(ghr & (1 << i)) fileObj << "1,";
    else fileObj << "0,";
  }
  
  // local history
  for(int i = HIST_LEN-1; i >= 0; --i){
    if(lht[phtIndex] & (1 << i)) fileObj << "1,";
    else fileObj << "0,";
  }

  // bpu prediction output
  for(int i = 1; i >= 0; i--){
    if(phtCounter & (1 << i)) fileObj << "1,";
    else fileObj << "0,";
  }

  for(int i = 1; i >= 0; i--){
    if(lphtCounter & (1 << i)) fileObj << "1,";
    else fileObj << "0,";
  }

  for(int i = 1; i >= 0; i--){
    if(bimodal[phtIndex] & (1 << i)) fileObj << "1,";
    else fileObj << "0,";
  }

  return;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

/***********************************************************/
#endif

