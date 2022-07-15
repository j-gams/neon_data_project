### logger

class mdl_log:
    def __init__ (self, model_path, model_name, tod, hparams, tparams):
        ### model name (name for log)
        self.model_name = model_name
        ### model path (path where model is saved)
        self.model_path = model_path
        ### both dicts below
        self.model_hyperparameters = hparams
        self.training_parameters = tparams
        self.datamain = []
        self.time = tod
        metrics = tparams["metrics"]
        self.datamain.append(["header:", model_path, model_name])
        self.datamain.append(["colnames:", "fold"] +
                [met for met in metrics])
        self.datamain.append(["hparams:", hparams])
        self.datamain.append(["tparams:", tparams])
        self.folds_recorded = 0
        print("generating training log")
    
    def add_record (self, stats, fold):
        #if isinstance(fold, int):
        #    #on a validation fold
        #    fold = "fold_" + str(fold)
        tempappend = ["fold_" + str(fold) + ":", fold]
        for stat in stats:
            tempappend.append(stat)
        self.folds_recorded += 1
        self.datamain.append(tempappend)
        
    def log_record (self):
        ### do 
        ### - print summary,
        ### - human readable report
        ### - csv report
        ### include
        ### - test
        ### - each fold
        ### - average fold
        ### - parameter summary
        
        ### prepare print summary
        ### header in pos 0
        ### hparams in pos 1
        ### tparams in pos 2

        ### compute avg, max, min over train
        metrics = self.datamain[1][2:]
        avgs = []
        mins = []
        maxs = []
        foldmin = []
        foldmax = []
        n_in_avg = 0
        ### want to do min among just train folds
        for met in range(len(metrics)):
            avgs.append(0)
            mins.append(float("inf"))
            maxs.append(float("-inf"))
            foldmin.append(-1)
            foldmax.append(-1)
            for i in range(4, len(self.datamain)):
                if self.datamain[i][0] == "fold_test":
                    n_in_avg += 1
                avgs[-1].append(self.datamain[i][met+2])
                if self.datamain[i][met+2] < mins:
                    mins = self.datamain[i][met+2]
                    foldmin = i
                if self.datamain[i][met+2] > maxs:
                    maxs = self.datamain[i][met+2]
                    foldmax = i
        for met in range(len(metrics)):
            avgs /= n_in_avg

        supp_data = [["averages:", "-"] + avgs, 
                     ["minimums:", "-"] + mins, 
                     ["min fold:", "-"] + foldmin, 
                     ["maximums:", "-"] + maxs,
                     ["max fold:", "-"] + foldmax]

        ### done finding avgs, mins, maxs

        ### one header and one for each fold
        printarr = [[] for ii in range(self.folds_recorded + 1)]
        printarr[0] = ["data:", "fold"]
        printarr[0].append([met for met in metrics])
        for i in range(4, len(self.datamain)):
            ### figure out which fold
            if self.datamain[i][0] == "fold_test":
                ### append to front
                printarr[1] = self.datamain[i]
                #printarr[1].append([dm for dm in self.datamain[i][2:]])
            else:
                printarr[self.datamain[i][1]+2] = self.datamain[i]
                #printarr[self.datamain[i][1]+2].append(
                #        [dm for dm in self.datamain[i][2:]])
        ### only keep non-empty ones?
        for elt in printarr:
            if elt == []:
                del elt
        ### print human readable
        txt_out = open("../logs/summary_" + self.model_name + "_" + 
                self.time + ".txt", "w+")
        disp_str = ["* MODEL SUMMARY"]
        disp_str += self.readable_arrange(self.datamain[0], "l1a")
        disp_str += self.readable_arrange(printarr, "l")
        disp_str += self.readable_arrange(supp_data, "l")
        disp_str += ["* TRAINED WITH HYPERPARAMETERS"]
        disp_str += ["hparams:"]
        disp_str += self.readable_arrange(self.datamain[2], "d") 
        disp_str += ["* TRAINED WITH PARAMETERS"]
        disp_str += ["tparams:"]
        disp_str += self.readable_arrange(self.datamain[3], "d")
        for lineout in disp_str:
            print(lineout)
            txt_out.write(lineout)
        print("done writing readable log")

    def readable_arrange(self, p_item, mode, frontpad=""):
        ### dict mode
        if mode == "d":
            klist = []
            kmaxlength = 0
            linelist = []
            for k in p_item:
                klist.append(str(k))
                if len(klist[-1]) > kmaxlength:
                    kmaxlength = len(klist[-1])
            colmaxlength = []
            for k in klist:
                linelist.append(str(p_item[k]))
                t_str = k + ":"
                while len(t_str) < kmaxlength + 3:
                    t_str += " "
                t_str = frontpad + t_str
                linelist[-1] = t_str + linelist[-1]
            return linelist
        ### 2d list mode
        elif mode == "l2":
            maxcollen = []
            linelist = []
            for i in range(len(p_item[0])):
                maxcollen.append(0)
                for j in range(len(p_item)):
                    if len(str(p_item[j][i])) > maxcollen[i]:
                        maxcollen[i] = len(str(p_item[j][i]))
            ### now make strings for linelist
            ### buffer is 2
            for i in range(len(p_item)):
                tstr = frontpad
                for j in range(len(p_item[i])):
                    tstr += str(p_item[i][j])
                    for k in range(maxcollen[i] - len(str(p_item[i][j])) + 2):
                        tstr += " "
                linelist.append(tstr)
            return linelist
        elif mode == "l1a":
            linelist = [frontpad]
            for i in range(len(p_item)):
                linelist[0] += str(p_item[i])
                linelist[0] += "  "
            linelist[0] = linelist[0][:-2]
            return linelist

        elif mode == "l1b":
            linelist = []
            for i in range(len(p_item))
                linelist.append(frontpad + str(p_item[i]))
            return linelist

    def machine_arrange(self):
        pass


