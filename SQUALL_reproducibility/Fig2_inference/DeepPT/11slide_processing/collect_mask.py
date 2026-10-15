import os,sys
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger, PdfFileReader
##======================================================================================================================
project = sys.argv[1] 

path2meta = "../10metadata/"
path2inputs = f"{project}_mask/"
path2outputs = ""

## find nume files within a folder
slide_names = []
for f in os.listdir(path2inputs):
    if f.endswith(".pdf"):
        slide_names.append(f)

## alphabet sort
slide_names = sorted(slide_names)


mergedObject = PdfFileMerger()
for slide_name in slide_names:    
    mergedObject.append(PdfFileReader("%s%s"%(path2inputs, slide_name), "rb"))
    
mergedObject.write("%s_mask.pdf"%project)

print("--- completed collecting mask--- ")
##-----------------------------------
df = pd.read_csv(f"{path2meta}{project}_slide_matched.csv")

slide_names_all = df["slide_name"].values

slide_names_short = np.array([x[:-4] for x in slide_names])

slide_names_missing = np.setdiff1d(slide_names_all, slide_names_short)
print("slide_names_missing:", slide_names_missing)
print(slide_names_missing.shape)


print("--- completed all taks --- ")











