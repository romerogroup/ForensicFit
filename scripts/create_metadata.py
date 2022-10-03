#!/usr/bin/env python3

""" Searches through all directories, finds files matching the
extension, compares it excel file provided, and returns a json
style lookup table.
"""
__author__ = "Pedram Tavadze"
   



import pandas as pd
from pathlib import Path
import re
import json
import argparse
from typing import Union, Dict, List


def get_files(path: Path, 
              ret: list = {}, 
              ext: str = '.tif',
              avoid: list = []) -> List[Dict[str, Path]]:
    for x in path.iterdir():
        if x.is_file() and x.suffix == ext :
            ret[x.stem] = x
        elif x.is_dir() and x.stem not in avoid:
            get_files(x, ret, ext, avoid)
    return ret

def get_metadata(file_path: Path, key: str) -> Dict[str, str]:
    surface = re.findall('\(([a-zA-Z]*)\)', key)[0]
    if 'stretch' in file_path.as_posix().lower():
        stretched = True
    else:
        stretched = False            
    
    if 'hand torn' in file_path.as_posix().lower():
        separation_method = "Hand Torn"
    elif 'scissor cut' in file_path.as_posix().lower():
        separation_method = "Scissor Cut"
    
    if "medium quality" in file_path.as_posix().lower():
        quality = "Medium Quality"
    elif "high quality" in file_path.as_posix().lower():
        quality = "High Quality"
    elif "low quality" in file_path.as_posix().lower():
        quality = "Low Quality"

    if 'mod' in file_path.as_posix():
        modified = True
    else:
        modified = False
    filename = file_path.stem.replace('_mod','')
    
    return {'filename': filename,
            'surface': surface,
            'quality': quality,
            'separation_method': separation_method,
            'stretched': stretched,
            'modified': modified, 
            'source': file_path.absolute().as_posix(),
            }

def get_lookup(files_dict: Dict[str, str], 
               df: pd.DataFrame) -> List[Dict]:
    """generates a lookup table for metadata

    Parameters
    ----------
    files_dict : Dict[str, str]
        Dictionary containing metadata about all files found
    df : pd.DataFrame
        Match or non-match excel file

    Returns
    -------
    List[Dict]
        list of all the files that have been mentioned in the pandas
        dataframe provided and their metadata


    """
    lookup = []
    not_exists = []
    for entry in df.to_dict('records'):
        for key in entry:
            if 'Tape' in key:
                name = entry[key][:-2]
                if name in files_dict:
                    file_path = files_dict[name]
                if name+'_mod' in files_dict:
                    file_path = files_dict[name+'_mod']
                lookup.append(get_metadata(file_path, key))    
                if name not in files_dict:
                    not_exists.append(entry[key])
    return lookup, not_exists
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        "Searches through all directories, finds files matching the"
        "extension, compares it excel file provided, and returns a json"
        "style lookup table.")
                                     )
    parser.add_argument('-d', '--dpath', 
                        dest='path',
                        required=True,
                        help='Directory to tree search')
    parser.add_argument('-a', '--avoid', 
                        dest='avoid', 
                        nargs='+', 
                        help='Directories to avoid searching',
                        default=[])
    parser.add_argument('-e','--excel-files',
                        dest='path_excel',
                        required=True,
                        nargs='+')
    parser.add_argument('-x','--extension',
                        dest='ext',
                        default='.tif',
                        help='Extension of the files being searched for')
    parser.add_argument('-o', '--output',
                        dest='output',
                        type=str,
                        default='metadata.json',
                        help='Output (metadata) filename',
                        )
    args = parser.parse_args()
    
    file_dict = get_files(Path(args.path), ext=args.ext, avoid=args.avoid)
    dfs = [pd.read_excel(x, engine='openpyxl') for x in args.path_excel]
    df = pd.concat(dfs)
    cols = [x for x in df.columns if 'Tape' in x or 'Rotation' in x]
    df = df[cols]
    del dfs
    lookup, not_exists = get_lookup(file_dict, df)
    print("-------------------------------------------------")
    print('The following files do not exist on this storage.')
    for x in not_exists:
        print(x)
    with open(args.output,'w') as wf:
        json.dump(lookup, wf, indent=2)
