import sys
import numpy as np

file_path = 'input_sample_ids.txt'


batches = {'Batch1':{
                '1Day':['28','30','30Baffle','32','32Baffle'],
                '2Day':['28','30','30Baffle','32','32Baffle'],
                'Salt':['28','30','30Baffle','32','32Baffle']
                },
      'Batch2':{
                '1Day':['1.4','1.4Baffle','1.5','1.8','1.8Baffle'],
                '2Day':['1.4','1.4Baffle','1.5','1.8','1.8Baffle'],
                'Salt':['1.4','1.4Baffle','1.5','1.8','1.8Baffle']
                },
       'Batch3':{
                '1Day':['M3','NS','Suc','SucBaffle','Raf', 'RafBaffle'],
                '2Day':['M3','NS','Suc','SucBaffle','Raf', 'RafBaffle'],
                'Salt':['M3','NS','Suc','SucBaffle','Raf', 'RafBaffle']
                },
       'Batch4':{
                '1Day':['M3','4uM_H89','10uM_H89','10uM_Paromomycin','50uM_Paromomycin', '0.5%EtOH'],
                '2Day':['M3','4uM_H89','10uM_H89','10uM_Paromomycin','50uM_Paromomycin', '0.5%EtOH'],
                'Salt':['M3','4uM_H89','10uM_H89','10uM_Paromomycin','50uM_Paromomycin', '0.5%EtOH']
                }
      }

replicates = [1,2]
counter=0
with open(file_path, 'w') as f:
    for batch,conditions in batches.items():
            for home_condition,perturbations in conditions.items():
                for perturbation in perturbations:
                    for rep in replicates:
                        f.write(f'{batch}\t{home_condition}\t{perturbation}\t{rep}\tbc_counts.csv\n')
                        counter+=1

print(counter)