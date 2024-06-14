from predictor import BASEPREDICTOR, UINT64, OpType
import xgboost as xgb
import numpy as np

DEBUG = False

# noinspection PyPep8Naming
class PREDICTOR(BASEPREDICTOR):
    """A gshare predictor from CBP-16."""
    #__slots__ = () # uncommet this to prevent seg fault in c++
    
    def __init__(self):
        self.history_len = 17
        self.ctr_max = 3
        self.global_history = 0
        self.num_entries = 2 ** 17
        self.gshare_table = [2] * (2 ** 17)
        self.local_history_table = [2] * (2 ** 17)
        self.local_history_pattern = [0] * (2 ** 17)
        self.selector_table = [2] * (2 ** 17)
        # change here to evaluate different trained models
        self.model_name = "src/simpython/random_forest_models/xgb_model_ALL.json"
        self.model = xgb.Booster()
        self.model.load_model(self.model_name)

    def sat_increment(self,x):
        if(x < self.ctr_max): 
            return x+1
        else: 
            return x
    
    def sat_decrement(self,x):
        if(x > 0): 
            return x-1
        else: 
            return x

    # bool GetPrediction(UINT64 PC)
    def GetPrediction(self,
                      PC: UINT64) -> bool:
        if DEBUG:
            print('GetPrediction | PC =', PC)

        index = ((PC >> 2) ^ self.global_history) % self.num_entries
        lht_index = self.local_history_pattern[index] % self.num_entries
        pht_pred = self.gshare_table[index]
        lht_pred = self.local_history_table[lht_index]
        selector_pred = self.selector_table[index]

        # machine learning model input
        input = []
        histories = [self.global_history,self.local_history_pattern[index]]
        for i in range(len(histories)):
            for j in range(self.history_len-1,-1,-1):
                if(histories[i] & (1<<j)):
                    input.append(1)
                else:
                    input.append(0)
        
        preds = [pht_pred,lht_pred,selector_pred]
        for i in range(len(preds)):
            for j in range(1,-1,-1):
                if(preds[i] & (1<<j)):
                    input.append(1)
                else:
                    input.append(0)

        # model predict
        input = np.array(input)
        input = input[np.newaxis,:]   
        input = xgb.DMatrix(input)     
        ml_pred = (self.model.predict(input) > 0.5).astype("int32")
        return ml_pred


        # original hybrid predictor prediction output
        """
        if(selector_pred & 0b10):
            if(lht_pred & 0b10): 
                return True
            else: 
                return False
        else:
            if (pht_pred & 0b10): 
                return True 
            else:
                return False
        """

    # void UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir,
    #                      bool predDir, UINT64 branchTarget)
    def UpdatePredictor(self,
                        PC: UINT64,
                        opType: OpType,
                        resolveDir: bool,
                        predDir: bool,
                        branchTarget: UINT64):
        if DEBUG:
            print(
                'UpdatePredictor | PC = {} OpType = {} resolveDir = {} '
                'predDir = {} branchTarget = {}'.format(
                    PC, opType, resolveDir, predDir, branchTarget)
            )

        index = ((PC >> 2) ^ self.global_history) % self.num_entries
        lht_index = self.local_history_pattern[index] % self.num_entries
        pht_pred = self.gshare_table[index]
        lht_pred = self.local_history_table[lht_index]
        selector_pred = self.selector_table[index]

        pht_taken = (pht_pred > (self.ctr_max // 2))
        lht_taken = (lht_pred > (self.ctr_max // 2))

        if(pht_taken != lht_taken):
            if(resolveDir == lht_taken):
                self.selector_table[index] = self.sat_increment(selector_pred)
            else:
                self.selector_table[index] = self.sat_decrement(selector_pred)

        if(resolveDir):
            self.gshare_table[index] = self.sat_increment(pht_pred)
            self.local_history_table[lht_index] = self.sat_increment(lht_pred)
        else:
            self.gshare_table[index] = self.sat_decrement(pht_pred)
            self.local_history_table[lht_index] = self.sat_decrement(lht_pred)

        self.HistoryUpdate(PC,opType,resolveDir,branchTarget)


    # void HistoryUpdate(UINT64 PC, OpType opType, bool resolveDir, 
    #                     UINT64 branchTarget)
    def HistoryUpdate(self,
                       PC: UINT64,
                       opType: OpType,
                       resolveDir: bool,
                       branchTarget: UINT64):
        
        if DEBUG:
            print(
                'HistoryUpdate | PC = {} OpType = {} '
                'taken = {} branchTarget = {}'.format(
                    PC, opType, resolveDir, branchTarget)
            )

        index = ((PC >> 2) ^ self.global_history) % self.num_entries
        self.global_history <<= 1
        self.local_history_pattern[index] <<= 1
        
        #print("in update, index = {}".format(index))
        if(resolveDir): 
            self.global_history += 1
            self.local_history_pattern[index] += 1

        self.global_history %= self.num_entries
        self.local_history_pattern[index] %= self.num_entries

    # void TrackOtherInst(UINT64 PC, OpType opType, bool taken,
    #                     UINT64 branchTarget)
    def TrackOtherInst(self,
                       PC: UINT64,
                       opType: OpType,
                       taken: bool,
                       branchTarget: UINT64):
        if DEBUG:
            print(
                'UpdatePredictor | PC = {} OpType = {} '
                'taken = {} branchTarget = {}'.format(
                    PC, opType, taken, branchTarget)
            )
        
        self.HistoryUpdate(PC,opType,taken,branchTarget)
        

