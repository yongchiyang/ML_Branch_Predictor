import matplotlib.pyplot as plt
import numpy as np

#types = ["SHORT_MOBILE","SHORT_SERVER","LONG_SERVER","LONG_MOBILE","ALL"]
#types = ["SHORT_MOBILE","SHORT_SERVER","LONG_SERVER"]
types = ["SHORT_MOBILE","SHORT_SERVER","LONG_SERVER","LONG_MOBILE"]


for type in types:
    total_icnt,total_branch = [],[]
    HYB_mispred,HYB_mpki,HYB_acc = [],[],[]
    TAGE_mispred,TAGE_mpki,TAGE_acc = [],[],[]
    MLP_mispred,MLP_mpki,MLP_acc = [],[],[]


    with open("{}_test_info".format(type), "r") as f:
        for line in f:
            parts = line.strip().split(',')
            total_icnt.append(int(parts[1]))
            HYB_mispred.append(int(parts[2]))
            HYB_mpki.append(float(parts[3]))

    # use ALL{}_mispredict to evaluate ALL.keras model
    with open("{}_mispredict".format(type),"r") as f:
        for line in f:
            parts = line.strip().split(',')
            total_branch.append(int(parts[1]))
            MLP_mispred.append(int(parts[2]))

    with open("{}_TAGE".format(type),"r") as f:
        for line in f:
            parts = line.strip().split(',')
            TAGE_mispred.append(int(parts[2]))
            TAGE_mpki.append(float(parts[3]))


    for total, misp in zip(total_icnt,MLP_mispred):
        MLP_mpki.append(misp * 1000 / total)

    for hyb_misp,test_misp,tage_misp,total in zip(HYB_mispred,MLP_mispred,TAGE_mispred,total_branch):
        HYB_acc.append(1-(hyb_misp/total))
        MLP_acc.append(1-(test_misp/total))
        TAGE_acc.append(1-(tage_misp/total))
        
    baseline = np.arange(1,len(total_icnt)+1)
    plt.plot(baseline,np.array(HYB_mispred),baseline,np.array(MLP_mispred),baseline,np.array(TAGE_mispred))
    plt.title('mispredict')
    plt.legend(('hybrid','mlp','tage'))
    plt.savefig("{}-mispred.png".format(type))

    plt.clf()

    plt.plot(baseline,np.array(HYB_mpki),baseline,np.array(MLP_mpki),baseline,np.array(TAGE_mpki))
    plt.title('mpki')
    plt.legend(('hybrid','mlp','tage'))
    plt.savefig("{}-mpki.png".format(type))

    plt.clf()
    plt.plot(baseline,np.array(HYB_acc),baseline,np.array(MLP_acc),baseline,np.array(TAGE_acc))
    plt.title('acc')
    plt.legend(('hybrid','mlp','tage'))
    plt.savefig("{}-accuracy.png".format(type))

    
    with open("{}".format(type),"a") as f:
        f.write("traces,total_instruction_cnt,total_branch_cnt,TAGE_mispred,HYB_mispred,ML_mispred,TAGE_mpki,HYB_mpki,ML_mpki,TAGE_acc,HYB_acc,ML_acc\n")
        for i in range(len(total_branch)):
            f.write("{}-{},{},{},{},{},{},{},{},{},{},{},{}\n".format(type,i+1,total_icnt[i],total_branch[i],TAGE_mispred[i],HYB_mispred[i],MLP_mispred[i],TAGE_mpki[i],HYB_mpki[i],MLP_mpki[i],TAGE_acc[i],HYB_acc[i],MLP_acc[i]))
        f.write("mean:\n")
        f.write("TAGE mpki = {}\n".format(sum(TAGE_mpki)/len(TAGE_mpki)))
        f.write("HYB mpki = {}\n".format(sum(HYB_mpki)/len(HYB_mpki)))
        f.write("MLP mpki = {}\n".format(sum(MLP_mpki)/len(MLP_mpki)))
        f.write("TAGE acc = {}\n".format(sum(TAGE_acc)/len(TAGE_acc)))
        f.write("HYB acc = {}\n".format(sum(HYB_acc)/len(HYB_acc)))
        f.write("MLP acc = {}\n".format(sum(MLP_acc)/len(MLP_acc)))
    
    #print("mispred, mpki, acc")
    #for i in range(1,len(total_icnt)+1):    
    #    print("trace {}, bpu :{}, {}, {:.5f}\nmlp :{}, {:.5f}, {:.5f}".format(i,mispred[i-1],mpki[i-1],acc[i-1],MLP_mispred[i-1],MLP_mpki[i-1],MLP_acc[i-1]))