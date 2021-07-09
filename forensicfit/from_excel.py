# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import pandas as pd
from numpy import array
from .core import Tape, TapeAnalyzer
from .core import Data
from .database import Database



def exists(db, filename, side="R", flip_h=False, analysis_mode="coordinate_based"):

    return db.gridfs_analysis.exists(
        {
            "$and": [
                {"filename": filename},
                {"metadata.side": side},
                {"metadata.image.flip_h": flip_h},
                {"metadata.analysis_mode": analysis_mode},
            ]
        }
    )




def from_excel(
    excel_file,
    modes=["coordinate_based", "weft_based", "big_picture", "max_contrast"],
    db_name="forensicfit",
    host="localhost",
    port=27017,
    username="",
    password="",
    
):

    db = Database(db_name, host, port, username, password)

    df = pd.read_excel(excel_file)
    ret = {key:{"data":[],"label":[]} for key in modes}
    ndata = len(df)
    for ientry in tqdm(range(ndata)):
        query = []
        for isurface in ["f", "b"]:
            for itape in [1, 2]:
                name = df.iloc[ientry]["tape_{}{}".format(isurface, itape)] + ".tif"
                side = df.iloc[ientry]["side_{}{}".format(isurface, itape)]
                if itape == 2:
                    flip_h = bool(df.iloc[ientry]["flip_{}".format(isurface)])
                else:
                    flip_h = False
                all_exists = True
                for imode in modes:
                    if not exists(
                        db, name, side=side, flip_h=flip_h, analysis_mode=imode
                    ):
                        all_exists = False
                if all_exists:
                    query.append(db.get_analysis(filename=name,side=side, flip_h=flip_h))
                else :
                    print("Not in the database:", name, side)
        if len(query) == 4:
            for imode in modes:
                if len(query[0][imode].shape) == 3:
                    for j in range(query[0][imode].shape[0]):
                        temp = []
                        for i in range(4):
                            temp.append(query[i][imode][j])
                        ret[imode]['data'].append(temp)
                        ret[imode]['label'].append(df.iloc[ientry]['match'])
                else :
                    ret[imode]['data'].append([x[imode] for x in query])
                    ret[imode]['label'].append(df.iloc[ientry]['match'])
                    
    for imode in modes:
        ret[imode] = Data(array(ret[imode]['data']),ret[imode]['label'])
    
    return ret
