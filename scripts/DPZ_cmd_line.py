# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 11:07:42 2025

@author: Adam Tejkl
"""

if __name__ == "__main__":
    import argparse
    import os
    import sys

    from datetime import date
    import time
    
    sys.path.append(r'\\data.fsv.cvut.cz\shares\K143\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\02_Tejkl')
    
    from DPZ_model_segment_raster import CMD_run

    parser = argparse.ArgumentParser(description="Run the trained model on raster data.")
    parser.add_argument("--root", required=True, help="Path to input raster file")
    parser.add_argument("--config_json_path", required=True, help="Path to save classified output")
    parser.add_argument("--model", required=False, default="model_final.keras", help="Path to trained model")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size for inference")

    args = parser.parse_args()
        
    # root = r'\\data.fsv.cvut.cz\shares\K143\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\04_plug_in'
    # config_json_path = r'\\data.fsv.cvut.cz\shares\K143\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\04_plug_in\config.json'

    print(f"   Loaded {args.root} and {args.config_json_path}")
    root = args.root
    config_json_path = args.config_json_path

    cmd_run = CMD_run(root, config_json_path)
    print("   Class CMD_run loaded")

    cmd_run.create_fresh_dbs()
    print("   Tool create_fresh_dbs run")

    cmd_run.fill_table_with_metadata()
    print("   Tool fill_table_with_metadata run")

    # category_list = ['Rill', 'NoRill']
    
    # for category in category_list:
    for shift in range(0,4):
        filename = cmd_run.fill_segmentation_setting(shift)
        print(f"   Tool fill_segmentation_setting run for shift {shift}")


    cmd_run.segment_and_classify()
    print("   Tool segment_and_classify run")
    
    for shift in range(0,4):
        classified_array = cmd_run.create_classified_array(shift, 'Rill')
        print(f"   Tool create_classified_array run for shift {shift} and Rill")
        
        classified_array = cmd_run.create_classified_array(shift, 'NoRill')
        print(f"   Tool create_classified_array run for shift {shift} and NoRill")
        
        
    averaged_rill_array = cmd_run.average_classified_arrays(filename, 'Rill')
    print("   Tool average_classified_arrays run for and Rill")
    filtered_Rill_array = cmd_run.filter_array(averaged_rill_array)
    print("   Tool filter_array run for Rill")
    
    averaged_NoRill_array = cmd_run.average_classified_arrays(filename, 'NoRill')
    print("   Tool average_classified_arrays run for NoRill")
    filtered_NoRill_array = cmd_run.filter_array(averaged_NoRill_array)
    print("   Tool filter_array run for NoRill")
    
    judged_array = cmd_run.judge_filtered_matrix(filtered_NoRill_array, filtered_Rill_array)
    print("   Tool judge_filtered_matrix")
        
    cmd_run.save_as_image(judged_array)
    print(f"Saving results to {args.output}")

    print("Done!")

    # check txt
    path = r'C:\Users\Adam Tejkl\Desktop'
    name = 'check_txt.txt'
    txt_path = os.path.join(path, name)
    
    with open(txt_path, "w") as f:
        f.write("Now the file has more content!")


#%%

#
##
#


