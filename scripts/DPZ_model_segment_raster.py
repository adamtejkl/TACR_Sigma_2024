# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:39:16 2025

@author: Adam Tejkl
"""

# import libraries
import os
import numpy as np
from PIL import Image
import json
import sqlite3
import pandas
import pandas as pd
import logging
import sys
import shutil

from datetime import date
import time

# import my modules
# sys.path.append(r'K:\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\02_Tejkl')
sys.path.append(r'\\data.fsv.cvut.cz\shares\K143\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\04_plug_in\scripts')
from DPZ_model import DB_management
from DPZ_debug_manager import Debug_Manager

# import special libraries

#%%

#
##
#

class Segmentation:

    def __init__(self, db_path, model_folder, model_number, config_json_path):
        """Initialize with the path to the database."""
        self.db_path = db_path
        self.model_folder = model_folder
        self.model_number = model_number
        # self.table_name = table_name

        self.mosaics_path = os.path.join(self.model_folder, "mosaics")

        # Path to the cancel txt
        self.cancel_txt_path = os.path.join(self.mosaics_path, "Cancel_model.txt")

        self.table_manag = DB_management(self.db_path)
        self.starting_db = Starting_database(self.db_path, self.model_folder, self.model_number, config_json_path)
        self.debug_manag = Debug_Manager(self.model_folder, self.db_path, config_json_path)

#%%

    def createTrainingMosaics_4(self, loc, storage_locations, class_csv, pixel_number, composite_array, shift, modification, just_others, shape):
        """
        Reads the class CSV file into a DataFrame and extracts class information.

        Parameters:
        class_csv (str): Path to the CSV file containing class information.
        shape (tuple): Tuple [width, height] of output mosaic

        Returns:
        tuple: A tuple containing lists of class names, class codes, and thresholds.
        """

        # verze "just_others" [ryhy, colormap]
        # verze "just_others" [colormap]
        A = 0

        ## raw composite array dimensions
        dimensions = composite_array.shape

        ## copy cell into new raster
        segment_area_pix = pixel_number * pixel_number   # number of pixels in cell

        i = 1
        for band in range(dimensions[0]-1):
            self.debug_manag.debug_action('print_all', print, f'   Band{i}')
    #         print(composite_array[i][4000:4020, 4000:4020])
    #         composite_array[i] = composite_array[i] / (1/255)
    #         composite_array[i] = composite_array[i].astype('int')
            i += 1

        ## treated array dimensions
        dimensions = composite_array.shape

    #     for band in range(dimensions[0]):
    #         print("    Band", i, end = ",")

        cancel = self.starting_db.check_for_cancel()
        if cancel == "Cancel":
            return("Cancel set to Yes")

        ## print informations
    #     print(cellSize, lowerLeft, dimensions, pixel_number, segment_area_pix, )

        ## set parameters based on just_others
        ## array maximums
        i = 0
        arr_max = []
        if just_others == "just_others":
            arr_max.append(np.amax(composite_array))
            y = dimensions[1]
            x = dimensions[2]
            start_band = 0
        else:
            y = dimensions[1]
            x = dimensions[2]
            start_band = 1
            for item in composite_array:
                if i == 0:
                    i += 1
                else:
                    arr_max.append(np.amax(composite_array[i]))
                    i += 1

        # print("   arr max:", arr_max)
        self.debug_manag.debug(f"   arr max: {arr_max}", statement_level=1)

        ## create log file
        log = [[loc, pixel_number, dimensions, shift]]
        self.debug_manag.debug(f"   log 0 line: {log[0]}", statement_level=1, end = "")

        # calculate for loop
        x_steps = int(x/pixel_number)
        y_steps = int(y/pixel_number)

        if x_steps*pixel_number > x:
            x_steps = x_steps - 1

        if y_steps*pixel_number > y:
            y_steps = y_steps - 1

        if shift == 1:
            x_steps = x_steps - 1
            y_steps = y_steps - 1
        elif shift == 2:
            y_steps = y_steps - 1
        elif shift == 3:
            x_steps = x_steps - 1
        else:
            pass

        for_range = x_steps * y_steps
        # print('   x steps:', x_steps,', y steps:', y_steps, ', For range ', for_range, ', dimensions[0]:', dimensions[0], end = "")
        self.debug_manag.debug(f"   x steps: {x_steps}, y steps: {y_steps}, For range: {for_range}, dimensions[0]: {dimensions[0]}", statement_level=1, end = "")

        A = 0
        percento = 0
        switch = 0
    #     for_range = 80

        width = shape[0] # of the mosaic
        height = shape[1] # of the mosaic

        ## save metadata if modification is none
        if modification == 'none':
            unit_spec_json_name = os.path.join(storage_locations[0], 'MTD_' + loc + '_' + str(shift) +'.json')
            mtd_dict = {'Location': loc,
                        'Pixel Number': pixel_number,
                        'Dimension': dimensions,
                        'Shift': shift,
                        'Width': width,
                        'Height': height}

            with open(unit_spec_json_name, "w") as write_file:
                json.dump(mtd_dict, write_file, indent=4)

        i = 0
        x = 0
        y = 0

        # reads class csv and returns lists
        class_names, class_codes, valid_codes, thresholds = self.read_class_csv(class_csv)

        noPix_threshold_ratio = 90.0

        noPix_count = 0
        # others_count = 0

    #         print(shift, k, l)
        # print("   CSV characterization ", len(class_names), len(class_codes[0]), len(valid_codes[0]))
        # print("   CSV characterization ", len(class_names), class_codes[0], class_codes[1])
        # print("   ", end = "")
        self.debug_manag.debug(f"""   CSV characterization, number of class names{len(class_names)}, starting class codes{class_codes[0]}, 
                             ending class codes {class_codes[1]}""", statement_level=1)
        self.debug_manag.debug("   ", statement_level=1, end = "")

        types_clause = ','.join([f' {typ}' for typ in class_names])
        value_clause = ','.join([' ?' for typ in class_codes[0]])  + ''
        # print(types_clause)
        # print(value_clause)

        # create db for segmented list
        fileName = f'List_{loc}_{shift}_log.db'
        output_file_name = os.path.join(storage_locations[0], fileName)

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function createTrainingMosaics_4 run for {loc} with shift {shift} and modification {modification}'
        log['Additional_log'] = f'Width {width}, Height {height}, threshold {thresholds[0]}, noPix_threshold_ratio {noPix_threshold_ratio}'
        log['Hierarchy'] = 2
        self.table_manag.create_record_from_dict('Processing_log', log)

        conn, cursor = self.setup_sqlite_db(output_file_name, class_names)

        # recognize shift and set k an l
        k, l = self.recognize_shift(shift, pixel_number)

        # early exit for testing
        # return

        for step in range(for_range):
            try:

                # control cancel txt"
                cancel = self.starting_db.check_for_cancel()
                if cancel == "Cancel":
                    break

                # clip segment of each band and assemble it into mosaic
                try:
                    mosaic_array = self.create_mosaic(start_band, dimensions[0], composite_array, k, l, pixel_number, modification, width, height)
                except Exception as e:
                    print(f"   Error in create_mosaic loop, step: {step}, k: {k}, l: {l}, ", str(e))

                # create counter for classifycation of the state of the mosaic
                states = []
                for item in class_codes[0]:
                    states.append(0)
                # state_noPix = 0
                state = "0"

                if just_others == "just_others":
                    # count pixels in mosaic based on affinity to edge
                    edge_array = composite_array[0][k:k+pixel_number, l:l+pixel_number]
                    # edge_array = composite_array[k:k+pixel_number, l:l+pixel_number] # for 1 band composites
                    values, counts = np.unique(edge_array, return_counts=True)

                    # for item in zip(values, counts):
                    #     if item[0] == 0.0 or item[0] == 256.0:
                    #         state_noPix += item(1)

                    state_noPix = self.noPix_routine(values, counts)

                    # add suffix based on percentage of pixel types
                    if state_noPix > (noPix_threshold_ratio/100.0)*segment_area_pix:
                        state = "99"
                    else:
                        state = "0"

                else:
                    # look for symbol signs
                    # count pixels in masaic based on type
                    symbol_array = composite_array[0][k:k+pixel_number, l:l+pixel_number]
                    values, counts = np.unique(symbol_array, return_counts=True)

                    # debugging part
        #             if len(values) > 1:
        #                 print(values, counts, x, y)
        #             if x == 50 and y == 50:
        #                 print("Symbol")
        #                 print(symbol_array)
        #                 print(f"   state {state_noPix}, values {values}, counts {counts}, state {state}")
        #                 print(class_codes[0], class_codes[1])

                    i = 0
                    for main_item in zip(class_codes[0], class_codes[1]):
                        for item in zip(values, counts):
                            if item[0] >= main_item[0] and item[0] <= main_item[1]:
                                states[i] += item[1]
                        i += 1

                    # count pixels in mosaic based on affinity to edge
                    edge_array = composite_array[1][k:k+pixel_number, l:l+pixel_number]
                    values, counts = np.unique(edge_array, return_counts=True)

                    # for item in zip(values, counts):
                    #     if item[0] == 0.0 or item[0] == 256.0:
                    #         state_noPix += item[1]
                    state_noPix = self.noPix_routine(values, counts)

                    # add suffix based on percentage of pixel types
                    if state_noPix > (noPix_threshold_ratio/100.0)*segment_area_pix:
                        state = "99"
                    else:
                        for item in zip(class_codes[0], states, thresholds):
                            if item[1] > (item[2]/100.0)*segment_area_pix:
                                state = str(item[0])

                # if the segment is not noData then continue with saving of the mosaic
                if state != '99':

                    X = str(x)
                    Y = str(y)

                    for c in range(x_steps):
                        if len(X) < len(str(x_steps)):
                            X = "0" + X

                    for c in range(y_steps):
                        if len(Y) < len(str(y_steps)):
                            Y = "0" + Y

                    name = "Mosaic_" + loc + "_X" + str(X)  + "Y" + str(Y) + "_" + state + "_" + str(shift)

                    # save mosaic into folder
                    # keras takes only jpeg, png, bmp, gif
                    # save mosaic if it contains data
                    appendix = '.jpg'

        #             print(mosaic_array)
                    im = Image.fromarray(mosaic_array)
                    im = im.convert('RGB')

                    # functional check
        #             if state == 1:
        #                 print(mosaic_array)
        #                 check +=1

                    # debugging part
        #             if x == 50 and y == 50:
        #                 print("Mosaic")
        #                 print(mosaic_array)
        #                 print("Edge")
        #                 print(edge_array)


                    # save picture to particular folder
                    i = 0
                    for typ in class_codes[0]:
        #                 print(typ, state)
                        if str(typ) == state:
                            mosaic_save_path = os.path.join(storage_locations[i+1], name + appendix)
                            line = [mosaic_save_path, name + appendix, X, Y, state]
                            im.save(mosaic_save_path)    # save only training mosaics
                            break
                        else:
                            # save to others folder
                            mosaic_save_path = os.path.join(storage_locations[-1], name + appendix)
                            line = [mosaic_save_path, name + appendix, X, Y, state]
                            im.save(mosaic_save_path)    # comment if you want to save only training and validation mosaics
                        i += 1
        #             print(line)

                    # im.save(mosaic_save_path)

                    for typ in class_codes[0]:
                        line.append(0)  # 0 for every type
                    line.append(0)    # 0 for Done
                    # print(line)

                    conn = sqlite3.connect(output_file_name)
                    cursor = conn.cursor()
                    cursor.execute(f"INSERT INTO segments (Path, Name, X_col, Y_row, State, {types_clause}, Done) VALUES (?, ?, ?, ?, ?,{value_clause}, ?)", line)
                    conn.commit()
                    conn.close()

                else:
                    noPix_count += 1

            except Exception as e:
                print(f"   Error in main segmentation loop, step: {step}, k: {k}, l: {l}, ", str(e))

            x += 1
            # s = 0
            # t = 0
            l = l + pixel_number

            # jump to new line
            if x == x_steps:
                x = 0
                l = 0
                k = k + pixel_number
                y += 1

                # recognize shift
                if shift == 1 or shift == 3:
                    l = int(pixel_number/2)
                else:
                    pass

    #         if state != "5":
    #             A += 1
    #
    #        early end
    #         if A == 5:
    #             break

            percento_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]

            if i == 10:
                print("   ", end = "")

            if round(A/for_range *100, 0) in percento_list and switch == 0:
                percento += 10
                switch = 1
                print(str(percento), "%, ", end = "")
            elif round(A/for_range *100, 0) not in percento_list:
                switch = 0
            else:
                pass

            A += 1

        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function createTrainingMosaics_4 run for {loc} finished'
        log['Hierarchy'] = 2
        self.table_manag.create_record_from_dict('Processing_log', log)

    #     print()
        # print("   Mosaic ", loc, ' shift ', shift, ' done')
        self.debug_manag.debug(f"   Mosaic: {loc}, shift: {shift} done", statement_level= 1)
    #     print('Check', check)
    #     print()

        return([fileName, output_file_name])

    #%%

    def noPix_routine(self, values, counts):

        state_noPix = 0
        for item in zip(values, counts):
            if item[0] == 0.0 or item[0] == 256.0:
                state_noPix += item[1]

        return(state_noPix)

    #%%

    def read_class_csv(self, class_csv):
        """
        Reads the class CSV file into a DataFrame and extracts class information.

        Parameters:
        class_csv (str): Path to the CSV file containing class information.

        Returns:
        tuple: A tuple containing lists of class names, class codes, and thresholds.
        """

        try:
            df = pd.read_csv(class_csv, sep=';')
            class_names = df['Class'].tolist()
            class_code_min = df['class_code_min'].tolist()
            class_code_max = df['class_code_max'].tolist()
            valid_code_min = df['valid_code_min'].tolist()
            valid_code_max = df['valid_code_max'].tolist()
            thresholds = df['thresholds'].tolist()
            logging.info("Class data loaded successfully from %s", class_csv)
            return class_names, [class_code_min, class_code_max], [valid_code_min, valid_code_max], thresholds
        except Exception as e:
            logging.error("    Error reading class CSV: %s", str(e))
            raise

    #%%

    def setup_sqlite_db(self, output_file_name, class_list):
        """
        Sets up the SQLite database for logging mosaic information.

        Parameters:
        outputFileName (str): Path and name of the database.
        class_list (list): List of class names.

        Returns:
        tuple: A tuple containing the SQLite connection and cursor.
        """
        try:
            conn = sqlite3.connect(output_file_name)
            cursor = conn.cursor()

            class_clause = ', '.join([f'{typ} INTEGER' for typ in class_list])
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS segments (
                Path TEXT,
                Name TEXT,
                X_col TEXT,
                Y_row TEXT,
                State TEXT,
                {class_clause},
                Done INTEGER
            )"""
            cursor.execute(create_table_query)
            logging.info("    SQLite database setup complete at %s", output_file_name)
            return conn, cursor
        except Exception as e:
            logging.error("    Error setting up SQLite database: %s", str(e))
            raise

    #%%

    def recognize_shift(self, shift, pixel_number):
        k, l = 0, 0

        if shift == 1:
            k = int(pixel_number/2)
            l = int(pixel_number/2)
        elif shift == 2:
            k = int(pixel_number/2)
        elif shift == 3:
            l= int(pixel_number/2)
        else:
            pass

        return (k, l)

    #%%

    def create_mosaic(self, start_band, end_band, composite_array, k, l, pixel_number, modification, width, height):
        """
        Creates mosaic of segments, width segments wide and height segments high
        Returns mosaic as an array.

        Parameters:
        start_band (int): Index of the starting band.
        end_band (int): Index of the last band.
        composite_array (numpy.ndarray): The multiband array to be mosaiced.
        k (int): Step in row direction.
        l (int): Step in column direction.
        pixel_number (int): Length of the segment side.
        modification (int): Type of the modification.
        width (int): Width of the created mosaic as a number of segments.
        height (int): Width of the created mosaic as a number of segments.
        """
        try:
            s, t, w, n  = 0, 0, 0, start_band
            mosaic_array = np.full((pixel_number * height, pixel_number * width), 0.0)
        # clip segment of each band and assemble it into mosaic
            for band in range(start_band, end_band):
                clip_array = composite_array[n][k:k+pixel_number, l:l+pixel_number]
        #             print(x, y, l, k, s, s + pixel_number, t, t + pixel_number, w, n, counter)
        #             print(clip_array)

        #             with np.nditer(clip_array, op_flags=['readwrite']) as it:
        #                 for pix in it:
        #                     pix[...] = pix * (256/arr_max[arr_count])

                if modification:
                    clip_array = self.apply_modifications(clip_array, modification)

                mosaic_array[t:t + pixel_number, s:s + pixel_number] = clip_array
                n += 1 # go on the next band
                s = s + pixel_number
                w += 1
                if w == width:
        #                 print("dosazeno w")
                    s = 0
                    w = 0
                    t = t + pixel_number
        #                 print(mosaic_array)
            return(mosaic_array)

        except Exception as e:
            logging.error("    Error creating mosaic: %s", str(e))
            raise

    #%%

    def save_mosaic(self, mosaic, path, name):
        """
        Saves the mosaic as an image file.

        Parameters:
        mosaic (numpy.ndarray): The mosaic array to be saved.
        path (str): Directory where the mosaic will be saved.
        name (str): Filename for the saved mosaic.
        """
        try:
            image = Image.fromarray(mosaic)
            image.save(os.path.join(path, name))
            logging.info("    Mosaic saved at %s", os.path.join(path, name))
        except Exception as e:
            logging.error("    Error saving mosaic: %s", str(e))
            raise

    #%%

    def log_mosaic_info(self, cursor, name, row, col, class_list, loc):
        """
        Logs information about the mosaic to the SQLite database.

        Parameters:
        cursor (sqlite3.Cursor): SQLite cursor object.
        name (str): Name of the mosaic.
        row (int): Row index of the mosaic.
        col (int): Column index of the mosaic.
        class_list (list): List of class names.
        loc (str): Location identifier.
        """
        try:
            cursor.execute(f"""
            INSERT INTO segments (Path, Name, X_col, Y_row, State, {', '.join(class_list)}, Done)
            VALUES (?, ?, ?, ?, ?, {', '.join(['0'] * len(class_list))}, 0)
            """, (loc, name, str(col), str(row), 'Pending'))
            logging.info("    Logged mosaic info for %s", name)
        except Exception as e:
            logging.error("    Error logging mosaic info: %s", str(e))
            raise

    #%%

    def apply_modifications(self, mosaic, modification):
        """
        Apply modifications to the mosaic such as rotation.

        Parameters:
        mosaic (numpy.ndarray): The mosaic array.
        modification (str): Type of modification to apply.

        Returns:
        numpy.ndarray: The modified mosaic array.
        """
        if modification == '90':
            return np.rot90(mosaic)
        elif modification == '180':
            return np.rot90(mosaic, k=2)
        elif modification == '270':
            return np.rot90(mosaic, k=3)
        elif modification == 'horizontal':
            return np.flipud(mosaic)
        elif modification == 'vertical':
            return np.fliplr(mosaic)
        else:
            pass

        return mosaic

    #%%



    def create_negative(self, mosaic):
        """
        Create negative version of the mosaic and save it

        Parameters:
        mosaic (numpy.ndarray): The mosaic array.

        Returns:
        numpy.ndarray: The modified mosaic array.
        """


    #%%

    def create_others_folder(self, folder, threshold):
        """
        Manages folder creation when the file count exceeds the threshold.

        Parameters:
        folder (str): Path to the model folder.
        threshold (int): File count threshold for creating a new folder.

        Returns:
        tuple: Updated others folder name.
        """
        try:
            # get others folders
            others_list = []
            folder_content = os.listdir(folder)
    #         print(folder_content)
            for item in folder_content:
                if "others" in item:
                    others_list.append(item)

            # get the highest others folder value
    #         print(others_list)
            max_others = 0
            for item in others_list:
                name_elements = item.split("_")
                if int(name_elements[-1]) > max_others:
                    max_others = int(name_elements[-1])

    #         print(max_others)

            # check for threshold
            file_count = 0
            all_files = os.listdir(os.path.join(folder, f"others_{max_others}"))
            file_count = len(all_files)
    #         print(file_count)

            if file_count > threshold:
                max_others += 1
                new_folder = f"others_{max_others}"
                os.mkdir(os.path.join(folder, new_folder))
                # print(f"   Threshold {threshold} overrun, new folder {new_folder} created")
                self.debug_manag.debug(f"   Threshold {threshold} overrun, new folder {new_folder} created", statement_level= 1)
                return(new_folder)

            else:
                new_folder = f"others_{max_others}"
                # print(f"   Threshold {threshold} not overrun, {file_count} files in original folder {new_folder}")
                self.debug_manag.debug(f"   Threshold {threshold} not overrun, {file_count} files in original folder {new_folder}", statement_level= 1)
                return(f"others_{max_others}")

        except Exception as e:
            logging.error("    Error in create_others_folder: %s", str(e))
            raise


    #%%

    def run_segmentation(self, folder, loc, class_csv, pixel_number, composite_array, shift, modification, just_others, shape):

        """
        Manages folder creation when the file count exceeds the threshold.

        Parameters:
        folder (str): Path to the model folder.
        loc (str): Base name for db, metadata and individual jpg.
        class_csv (str): Path to the csv with class setting.
        pixel_number (int): Segment edge length.
        composite_raster (raster): Compositer raster to segment
        shift (int): 0, 1, 2, 3 type of segment shift to be used.
        modification (str): 'none', '90', '180', '270', 'horizontal', 'vertical' type of segment modification to be used.
        just_others (bool): Flag to indicate creation of only "others" class.
        """
        try:

            # Read the CSV file into a DataFrame
            df = pd.read_csv(class_csv, sep = ";")

        except Exception as e:
            logging.error("   Error in reading class_csv: %s", str(e))
            raise

        try:

            # Extract class names, class codes, and thresholds as lists
            class_list = df['Class'].tolist()

        except Exception as e:
            logging.error("   Error in converting class_csv: %s into data frame", str(e))
            raise

        try:

            storage_locations = []
            storage_locations.append(self.mosaics_path)

            for item in class_list:
                path = os.path.join(self.mosaics_path, str(item))
                storage_locations.append(path)
                try:
                    os.mkdir(path)
                except:
                    pass

            path = os.path.join(folder, 'others_0')
            storage_locations.append(path)

        except Exception as e:
            logging.error("   Error in setting folder paths: %s", str(e))
            raise

        try:

            # check for overflowing others folder
            last_others = self.create_others_folder(folder, 800000)
            storage_locations[-1] = os.path.join(folder, last_others)

        except Exception as e:
            logging.error("   Error in create_others_folder: %s", str(e))
            raise

            # analyse it

        try:
            # print('   createTrainingMosaics_4 tool started')
            self.debug_manag.debug("   createTrainingMosaics_4 tool started", statement_level= 1)
            fileName, output_file_name = self.createTrainingMosaics_4(loc, storage_locations, class_csv, pixel_number, composite_array, shift, modification, just_others, shape)

        except Exception as e:
            logging.error("   Error in createTrainingMosaics_4: %s", str(e))
            raise

        return([fileName, output_file_name])

    #%%

    def get_processed_scenes(self, log_file):
        processed_successfully = []

        if os.path.exists(log_file):
            with open(log_file, 'r') as log_file:
                for line in log_file:
                    try:
                        parts = line.split(';')
                        filename = parts[1].strip()  # The second part is the filename
                        shift = parts[3].strip()
                        modification = parts[4].strip()
                        status = parts[5].strip()   # The last part is the segmentation status (successful or error)

                        new_log = [filename, shift, modification, status]

                    except:
                        print("except used")
                        pass

                    # Only skip scenes that were processed successfully
                    if status == "segmentation successfull":
                        processed_successfully.append(new_log)

        return processed_successfully


#%%

# model_number = "005"

# # model folder path
# folder = r"\\backup.fsv.cvut.cz\Users\tejklada\DPZ_junior_Tejkl\Models"
# model_folder = os.path.join(folder, f'model_{model_number}')
# raster_folder = r'\\data.fsv.cvut.cz\Projects\TACR-SIGMA-EROZE-ANN\cuzk_ortofoto\data_epsg5514'  # Or a geodatabase feature class

#%%

#
##
#

class Starting_database:

    def __init__(self, db_path, model_folder, model_number, config_json_path):
        """Initialize with the path to the database."""
        self.db_path = db_path
        self.model_folder = model_folder
        self.model_number = model_number
        self.config_json_path = config_json_path

        self.mosaics_path = os.path.join(self.model_folder, "mosaics")

        # Path to the cancel txt
        self.cancel_txt_path = os.path.join(self.mosaics_path, "Cancel_model.txt")

        self.table_manag = DB_management(self.db_path)
        self.debug_manag = Debug_Manager(self.model_folder, self.db_path, config_json_path)

        # get table dictionaries
        self.config_json_path = config_json_path
        self.paths_dict = self.table_manag.load_table_name_and_columns('json_paths', self.config_json_path)
        
        # get filetype and separator from config file
        config_data_dict = self.table_manag.load_table_name_and_columns('data', self.config_json_path)
        self.filetype = config_data_dict['filetype']
        self.separator = config_data_dict['separator']

#%%

#
## Create training table from freshly exported training rasters
#

    def create_database(self):

        # print(f"Started {date.today()}")
        self.debug_manag.debug(f"Started {date.today()}", statement_level= 0)

        i = 1

        # set db
        table_name = 'Segmentation_setting'
        table_columns = self.table_manag.load_table_name_and_columns('Segmentation_setting', os.path.join(self.root, self.paths_dict['tables']))
        train_dict =  self.table_manag.load_table_name_and_columns('train_dict', os.path.join(self.model_folder, self.paths_dict['tables']))

        # table_manag.drop_table(table_name)
        self.table_manag.create_table(table_name, table_columns)

        # create settings for segmentation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Iterate through the results
        ##class_csv = os.path.join(folder, "class_setting.csv")
        class_csv = os.path.join(self.model_folder, "class_setting_val.csv")
        pixel_number = 80
        just_others = ""

        # create dictionary for segmentation setting
        settings = {}
        settings['Folder'] = self.model_folder
        settings['Class_csv'] = class_csv
        settings['Pixel_number'] = pixel_number
        settings['Just_others'] = just_others
        settings['Status'] = "Segmentation setting created"

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function create_database run for {self.model_folder} and {self.model_number}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        i = 1

        # Execute query to select records where train_line_count is not 0
        cursor.execute("SELECT * FROM Lists WHERE train_line_count != 0")
        records = cursor.fetchall()
        # print(f"DB {self.db_path} scanned succesfully, {len(records)} rasters found")
        self.debug_manag.debug(f"DB {self.db_path} scanned succesfully, {len(records)} rasters found", statement_level= 1)

        cursor.execute("SELECT * FROM Lists WHERE train_line_count != 0")
        for row in cursor.fetchall():

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            # print(row[10])  # Process each row
            self.debug_manag.debug(f"   Row[10]: {row[10]}", statement_level= 2)
            list_dict = self.table_manag.retrieve_data_as_dict('Lists', 'List_ID', row[0], train_dict)
            settings['Filename'] = list_dict['List_Name']

            # model_folder = r"\\backup.fsv.cvut.cz\Users\tejklada\DPZ_junior_Tejkl\Models\model_005"   # Uncomment for changing the path
            train_folder = "training_rasters"
            list_name = list_dict['List_Name']
            new_list_name = list_name.replace("WRTO24_", "Train_")

            settings['Train_path'] = os.path.join(self.model_folder, train_folder, new_list_name)   # Uncomment for changing the ArcGIS path
            # settings['Train_path'] = list_dict['Train_path']   # Uncomment for keeping the ArcGIS path

            # Split raster to calibration polygons
            modifications = ['none']
            shifts = [0, 1, 2, 3]

            # creating the setting of individual segmentations

            for shift in shifts:
                for modification in modifications:

                    settings['Shift'] = shift
                    settings['Modification'] = modification

                    print('Shift ', shift, 'Modification ', modification, ' ', end = "")
                    self.table_manag.create_record_from_dict(table_name, settings)

            print()

            i += 1

        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = 'Function create_database finished'
        self.table_manag.create_record_from_dict('Processing_log', log)

        # Close the connection
        conn.close()


    #%%

    #
    ## Copy and treat database from already existing database
    #

    def create_nonArcGIS_database(self, models_folder, old_model_number):
        """
        Copy and treat database from already existing database

        :param model_folder: Path to the SQLite database file.
        :param model_number: Name of the table where the record will be updated.
        """

        print(f'Started by copying model model_{old_model_number}', date.today())

        # copy and paste the DB to new folder
        source = os.path.join(models_folder, f'model_{old_model_number}', f'model_{old_model_number}.db')
        destination = os.path.join(self.model_folder, f'model_{self.model_number}.db')
        shutil.copyfile(source, destination)
        print(f'DB {source} copied as {destination}')

        i = 1

        # set db
        train_dict =  self.table_manag.load_table_name_and_columns('train_dict', os.path.join(self.model_folder, self.paths_dict['tables']))

        # create settings for segmentation
        conn = sqlite3.connect(destination)
        cursor = conn.cursor()

        # create dictionary for updating Lists table
        update_dict = {}
        update_dict['Folder'] = self.model_folder

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function create_nonArcGIS_database run for {self.model_folder}, from model_{old_model_number} to model_{self.model_number}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        i = 1

        # Execute query to select records where train_line_count is not 0
        cursor.execute("SELECT * FROM Lists")
        records = cursor.fetchall()
        print(f"Db {destination} scanned succesfully, {len(records)} records found")

        cursor.execute("SELECT * FROM Lists")
        for row in cursor.fetchall():

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            print(row[10])  # Process each row
            list_dict = self.table_manag.retrieve_data_as_dict('Lists', 'List_ID', row[0], train_dict)

            list_name = list_dict['List_name']
            new_train_name = list_name.replace("WRTO24_", "Train_")

            update_dict['Train_path'] = list_dict['Train_path']
            # update_dict['Train_path'] = os.path.join(self.model_folder, 'training_rasters', new_train_name)   # Uncomment for changing the ArcGIS path

            # updating the line of the DB
            self.table_manag.update_record_from_dict('Lists', 'List_ID', row[0], update_dict)
            print(f"Folder set to {update_dict['Folder']}, Train_path set to {update_dict['Train_path']}")
            print()

            i += 1

        # drop Segmentation_setting table and create it again

        # set db
        table_name = 'Segmentation_setting'
        table_columns =  self.table_manag.load_table_name_and_columns('Segmentation_setting', os.path.join(self.model_folder, self.paths_dict['tables']))

        self.table_manag.drop_table(table_name)
        self.table_manag.create_table(table_name, table_columns)

        # create settings for segmentation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Iterate through the results
        ##class_csv = os.path.join(folder, "class_setting.csv")
        class_csv = os.path.join(self.model_folder, "class_setting_val.csv")
        pixel_number = 80
        just_others = ""

        # create dictionary for segmentation setting
        settings = {}
        settings['Folder'] = self.model_folder
        settings['Class_csv'] = class_csv
        settings['Pixel_number'] = pixel_number
        settings['Just_others'] = just_others
        settings['Status'] = "Segmentation setting created"

        # Execute query to select records where train_line_count is not 0
        cursor.execute("SELECT * FROM Lists WHERE train_line_count != 0")
        records = cursor.fetchall()
        print(f"DB {self.db_path} scanned succesfully, {len(records)} rasters found")

        i = 1

        cursor.execute("SELECT * FROM Lists WHERE train_line_count != 0")
        for row in cursor.fetchall():

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            print(row[10])  # Process each row
            list_dict = self.table_manag.retrieve_data_as_dict('Lists', 'List_ID', row[0], train_dict)
            settings['Filename'] = list_dict['List_name']

            # model_folder = r"\\backup.fsv.cvut.cz\Users\tejklada\DPZ_junior_Tejkl\Models\model_005"   # Uncomment for changing the path
            train_folder = "training_rasters"
            list_name = list_dict['List_name']
            new_list_name = list_name.replace("WRTO24_", "Train_")

            # settings['Train_path'] = os.path.join(self.model_folder, train_folder, new_list_name)   # Uncomment for changing the ArcGIS path
            settings['Train_path'] = list_dict['Train_path']   # Uncomment for keeping the ArcGIS path

            # Split raster to calibration polygons
            modifications = ['none']
            shifts = [0, 1, 2, 3]

            # creating the setting of individual segmentations

            for shift in shifts:
                for modification in modifications:

                    settings['Shift'] = shift
                    settings['Modification'] = modification

                    print('   Shift ', shift, 'Modification ', modification, ' ', end = "")
                    self.table_manag.create_record_from_dict(table_name, settings)

            print()

            i += 1

        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = 'Function create_nonArcGIS_database finished'
        self.table_manag.create_record_from_dict('Processing_log', log)

        # Close the connection
        conn.close()



    #%%

    #
    ## Create database from random folder with scenes
    #

    def create_classification_database(self, raster_folder, filetype):
        """
        Create a classification database for all raster files in a folder.

        This function scans the specified directory, finds all raster files
        matching the given filetype (e.g., ".jpg", ".tif"), and generates an
        SQLite classification database containing metadata needed for later
        processing and model-based classification.

        :param raster_folder:
            Path to the folder containing input raster files. Each file in this
            folder will be inspected and inserted into the classification database.

        :param filetype:
            File extension to filter raster files (e.g., ".jpg", ".tif").
            Only files with this extension will be processed.

        :return:
            None. The function creates or updates the SQLite database on disk.
        """

        print('Started', date.today())
        print()

        i = 1

        # set db
        train_dict = self.table_manag.load_table_name_and_columns('train_dict', os.path.join(self.model_folder, self.paths_dict['tables']))
        segmentation_columns = self.table_manag.load_table_name_and_columns('segment_dict', os.path.join(self.model_folder, self.paths_dict['tables']))
        processing_columns = self.table_manag.load_table_name_and_columns('processing_columns', os.path.join(self.model_folder, self.paths_dict['tables']))

        # table_manag.drop_table(table_name)
        self.table_manag.create_table('Lists', train_dict)
        self.table_manag.create_table('Segmentation_setting', segmentation_columns)
        self.table_manag.create_table('Processing_log', processing_columns)

        # Iterate through the results
        ##class_csv = os.path.join(folder, "class_setting.csv")
        class_csv = os.path.join(self.model_folder, "class_setting_val.csv")
        pixel_number = 80
        just_others = "just_others"

        # create dictionary for segmentation setting
        settings = {}
        settings['Folder'] = self.model_folder
        settings['Class_csv'] = class_csv
        settings['Pixel_number'] = pixel_number
        settings['Just_others'] = just_others
        settings['Status'] = "Segmentation setting created"

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function create_database run for {self.model_folder} and {self.model_number}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        i = 1

        # scan folder and get all scenes
        scenes = []
        content = os.listdir(raster_folder)
        # filter out only jpg
        for file in content:
            if file.lower().endswith(filetype):
                scenes.append(file)

        print(f"DB {raster_folder} scanned succesfully, {len(scenes)} rasters found")
        lists_settings = {}

        for scene in scenes:
            print(scene, end = "")
            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            lists_settings['List_Name'] = scene
            lists_settings['Folder'] =  self.model_folder
            lists_settings['Spatial_Reference'] = 'S-JTSK_Krovak_East_North'
            lists_settings['Train_status'] = 'No trainning lines'

            self.table_manag.create_record_from_dict('Lists', lists_settings)

            # Split raster to calibration polygons
            modifications = ['none']
            shifts = [0, 1, 2, 3]

            # creating the setting of individual segmentations

            for shift in shifts:
                for modification in modifications:

                    settings['Filename'] = scene
                    settings['Shift'] = shift
                    settings['Modification'] = modification

                    print('Shift ', shift, 'Modification ', modification, ' ', end = "")
                    self.table_manag.create_record_from_dict('Segmentation_setting', settings)

            print()

            i += 1

        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = 'Function create_database finished'
        self.table_manag.create_record_from_dict('Processing_log', log)

    #%%

    def segment_scenes(self, raster_folder, DMR_folder, shape, inverse = 0):
        """
        Create the specified table in the SQLite database using a dictionary of updated values.

        :param raster_folder:
        :param DMR_folder:
        :param shape (tuple): Tuple [width, height] of output mosaic
        """

        segment_dict = self.table_manag.load_table_name_and_columns('segment_dict', os.path.join(self.model_folder, self.paths_dict['tables']))

        # table_manag.create_table(table_name, table_columns)

        # create settings for segmentation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function segment_scenes run for {self.model_folder}, {raster_folder}, {DMR_folder} and model_{self.model_number}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        sql_query = "SELECT * FROM Segmentation_setting"

        cursor.execute(sql_query)
        records = cursor.fetchall()

        print(f"DB {self.db_path} scanned succesfully, {len(records)} rasters found")

        i, tot_time, C = 0, 0, 1

        if inverse == 1:
            records = list(reversed(records))

        for row in records:
            start = time.time()
            # print(row)

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            # stop after 5
            # if i == 5:
            #     break

            list_dict = self.table_manag.retrieve_data_as_dict('Segmentation_setting', 'Segmentation_ID', row[0], segment_dict)
            # print(list_dict)

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            # get status info
            print(f"   List {list_dict['Filename']}, status {list_dict['Status']}")

            # stop after 5
            # if i == 5:
            #     break

            # Get location
            loc, year = self.get_filename(list_dict['Filename'], self.filetype, self.separator)

            # Get class_db_name
            class_db_name = f"List_{loc}_{year}_{list_dict['Shift']}_log.db"
            class_db_path = os.path.join(list_dict['Folder'], 'mosaics', class_db_name)

            # image preprocessing
            if list_dict['Status'] == 'loading images error' or list_dict['Status'] == "Segmentation setting created" or list_dict['Status'] == 'Segmentation error':

                # composite_array, error_relay = self.image_loading_and_preprocessing_subroutine(list_dict, raster_folder)
                composite_array, error_relay = self.image_DMR_loading_and_preprocessing_subroutine(list_dict, raster_folder, DMR_folder)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Segmentation with shift {list_dict['Shift']} started at {round((end - start)/60, 2)} mins")
                class_db_name, class_db_path, error_relay = self.segmentation_subroutine(composite_array, list_dict, loc, shape)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

            elif list_dict['Status'] == 'Segmentation done':
                print(f"   List {list_dict['Filename']} finished")

            else:
                print(f"   List {list_dict['Filename']}, unknown status: {list_dict['Status']}")
                continue

            # time management and estimated time left
            i, C = self.est_time_left(i, start, tot_time, records, C, trigger = 5)


        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function segment_scenes finished and {i} scenes trated'
        self.table_manag.create_record_from_dict('Processing_log', log)

        conn.close()

        print(f"Processing of {i} scenes completed.")


    #%%

    #
    ## Segment scenes as just others and classify them at the same time, delete afterwards
    #

    def segment_and_classify_scenes(self, raster_folder, shape, model_path, inverse = 0):
        """
        Create the specified table in the SQLite database using a dictionary of updated values.

        :param raster_folder:
        :param DMR_folder:
        :param shape (tuple): Tuple [width, height] of output mosaic
        """

        segment_dict = self.table_manag.load_table_name_and_columns('segment_dict', os.path.join(self.model_folder, self.paths_dict['tables']))

        # table_manag.create_table(table_name, table_columns)

        # create settings for segmentation

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function segment_scenes run for {self.model_folder}, {raster_folder} and model_{self.model_number}, inverse {inverse}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql_query = "SELECT * FROM Segmentation_setting"

        cursor.execute(sql_query)
        records = cursor.fetchall()

        print(f"DB {self.db_path} scanned succesfully, {len(records)} rasters found, inverse {inverse}")

        i, tot_time, C = 0, 0, 1

        if inverse == 1:
            records = list(reversed(records))

        for row in records:
            start = time.time()

            # print(row)

            list_dict = self.table_manag.retrieve_data_as_dict('Segmentation_setting', 'Segmentation_ID', row[0], segment_dict)
            # print(list_dict)

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            # get status info
            print(f"   List {list_dict['Filename']}, status {list_dict['Status']}")

            # stop after 5
            # if i == 5:
            #     break

            # Get location
            loc, year = self.get_filename(list_dict['Filename'], self.filetype, self.separator)

            # Get class_db_name
            class_db_name = f"List_{loc}_{year}_{list_dict['Shift']}_log.db"
            class_db_path = os.path.join(list_dict['Folder'], 'mosaics', class_db_name)

            # image preprocessing
            if list_dict['Status'] == 'loading images error' or list_dict['Status'] == "Segmentation setting created":

                composite_array, error_relay = self.image_loading_and_preprocessing_subroutine(list_dict, raster_folder)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Segmentation with shift {list_dict['Shift']} started at {round((end - start)/60, 2)} mins")
                class_db_name, class_db_path, error_relay = self.segmentation_subroutine(composite_array, list_dict, loc, shape)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Classification started at {round((end - start)/60, 2)} mins")
                error_relay = self.classification_subroutine(list_dict, class_db_path, model_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                error_relay = self.deleting_subroutine(list_dict, class_db_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

            elif list_dict['Status'] == 'Segmentation error':

                composite_array, error_relay = self.image_loading_and_preprocessing_subroutine(list_dict, raster_folder)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Segmentation with shift {list_dict['Shift']} started at {round((end - start)/60, 2)} mins")
                class_db_name, class_db_path, error_relay = self.segmentation_subroutine(composite_array, list_dict, loc, shape)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Classification  started at {round((end - start)/60, 2)} mins")
                error_relay = self.classification_subroutine(list_dict, class_db_path, model_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Deleting started at {round((end - start)/60, 2)} mins")
                error_relay = self.deleting_subroutine(list_dict, class_db_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

            # classify segmented images
            elif list_dict['Status'] == 'Classification error' or list_dict['Status'] == 'Segmentation done':
                error_relay = self.classification_subroutine(list_dict, class_db_path, model_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

                end = time.time()
                print(f"   Deleting started at {round((end - start)/60, 2)} mins")
                error_relay = self.deleting_subroutine(list_dict, class_db_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

            # delete segmented images
            elif list_dict['Status'] == 'Deleting error' or list_dict['Status'] == 'Classification done' :
                error_relay = self.deleting_subroutine(list_dict, class_db_path)
                if error_relay == 1:
                    i += 1
                    continue  # Move on to the next scene even if there's an error

            elif list_dict['Status'] == 'Deleting done':
                print(f"   List {list_dict['Filename']} finished")

            else:
                print(f"   List {list_dict['Filename']}, unknown status: {list_dict['Status']}")
                break

            # time management and estimated time left
            i, C = self.est_time_left(i, start, tot_time, records, C, trigger = 5)


        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function segment_scenes finished and {i} scenes trated'
        self.table_manag.create_record_from_dict('Processing_log', log)

        conn.close()

        print(f"Processing of {i} scenes completed.")


    #%%

    #
    ## Individual segmenting soubroutines
    #

    # Image loading and preprocessing
    def image_loading_and_preprocessing_subroutine(self, list_dict, raster_folder):
        status_dict = {}

        try:
            # Full path to the RGB file
            RGB_path = os.path.join(raster_folder, list_dict['Filename'])

            # Open the RGB image
            RGB_img = Image.open(RGB_path)
            print(f'   RGB image {RGB_path} loaded')

            # Convert to NumPy array
            RGB_array = np.array(RGB_img)
            print('   RGB array shape ', RGB_array.shape, end = " ")
            print('   Images loaded as arrays')

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = 'loading images error'
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error loading images for {list_dict['Filename']}: {e}")
            composite_array, error_relay = 0, 1
            return [composite_array, error_relay]  # Move on to the next scene even if there's an error


        try:
            # create additional grayscale array and round it to nearest integer
            red, green, blue = 0, 1, 2
        #     grayscale_array = (0.3 * Red) + (0.59 * Green) + (0.11 * Blue)
        #     grayscale_array = ((0.3 * composite_array[red]) + (0.59 * composite_array[green]) + (0.11 * composite_array[blue]))
            grayscale_array = np.around((0.3 * RGB_array[:, :, red]) + (0.59 * RGB_array[:, :, green]) +  (0.11 * RGB_array[:, :, blue]), 0)
            grayscale_array = grayscale_array.astype('int')
            print('   Grayscale array shape ', grayscale_array.shape)
        #     print(grayscale_array[100:110, 100:110])

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Grayscale calculation error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error calculating grayscale {list_dict['Filename']}: {e}")
            composite_array, error_relay = 0, 1
            return [composite_array, error_relay]  # Move on to the next scene even if there's an error

        try:
            print("   Composite creation started 1")

            # Transpose RGB_array to (3, height, width)
            RGB_array = np.transpose(RGB_array, (2, 0, 1))

            # Reshape grayscale_array to (1, height, width)
            grayscale_array = np.expand_dims(grayscale_array, axis=0)

            # Now all arrays have the same (bands, height, width) format
            # print(f"   Train array shape: {train_array.shape}", end = " ")  # Expected: (1, 8000, 10000)
            # print(f"   RGB array shape: {RGB_array.shape}", end = " ")      # Expected: (3, 8000, 10000)
            # print(f"   Grayscale array shape: {grayscale_array.shape}", end = " ")  # Expected: (1, 8000, 10000)

            # Append along the first axis (bands)
            composite_array = np.concatenate((RGB_array, grayscale_array), axis=0)

            # Print final shape
            print(f"   Composite array shape: {composite_array.shape}", end = " ")  # Expected: (5, 8000, 10000)
            print("   Composite array created")

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Composite creation error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error compositing {list_dict['Filename']}: {e}")
            composite_array, error_relay = 0, 1
            return [composite_array, error_relay]  # Move on to the next scene even if there's an error

        error_relay = 0
        return [composite_array, error_relay]


    # Image loading, DMR loading and preprocessing subroutine
    def image_DMR_loading_and_preprocessing_subroutine(self, list_dict, raster_folder, DMR_folder):
        status_dict = {}

        try:
            # Open the training image
            train_img = Image.open(list_dict['Train_path'])
            print(f"   training image {list_dict['Train_path']} loaded")

            # Full path to the RGB file
            RGB_path = os.path.join(raster_folder, list_dict['Filename'])

            # Open the RGB image
            RGB_img = Image.open(RGB_path)
            print(f"   RGB image {RGB_path} loaded")

            # Get location
            loc, year = self.get_filename(list_dict['Filename'], self.filetype, self.separator)

            # # Full path to the DMR file
            DMR_path = os.path.join(DMR_folder, f"Flow_acc_DMR4g_{loc}")

            # # Open the dmr image
            DMR_img = Image.open(DMR_path)
            print(f"   DMR image {DMR_path} loaded")

            # Full path to the kapky file
            # name = loc.replace('.tif', '')
            # DMR_path = os.path.join(DMR_folder, f'{name}_5G_kapky.png')
            # DMR_path = os.path.join(DMR_folder, f'{name}_5G_slope.tif')

            # Open the dmr image
            # DMR_img = Image.open(DMR_path)
            # print(f'   DMR image {DMR_path} loaded')

            # Convert to NumPy array
            train_array = np.array(train_img)
            print('   Train array shape ', train_array.shape, end = " ")
            height, width = train_array.shape[0], train_array.shape[1]
            RGB_array = np.array(RGB_img)
            print('   RGB array shape ', RGB_array.shape, end = " ")

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = 'resizing DMR image error'
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error {status_dict['Status']} for {list_dict['Filename']}: {e}")



        try:
            # Resize to match the shape of Train array
            DMR_resized = DMR_img.resize((width, height), resample=Image.BILINEAR)
            DMR_array = np.array(DMR_resized)
            print('   DMR array shape ', DMR_array.shape, end = " ")

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "resizing DMR image error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error {status_dict['Status']} for {list_dict['Filename']}: {e}")

        try:
            # Normalize DMR to 0 to 255
            # print(type(DMR_array))
            # print(DMR_array[4000:4010, 5000:5010])

            # Replace any negative values with 0 (or np.nan if you want to mask them)
            DMR_array[DMR_array < 0] = 0
            DMR_array[DMR_array > 700] = 255

            DMR_array, DMR_min, DMR_max = self.normalize_array_to_uint8(DMR_array) # normalization of DMR arra
            # DMR_array = DMR_array[:, :, 0] # Keep only the first band of kapky
            print("   DMR array normalized shape ", DMR_array.shape, end = " ")
            print(f"   DMR_min: {DMR_min}, DMR_max: {DMR_max}", end = " ")
            # break

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = 'normalizing DMR image error'
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error {status_dict['Status']} for {list_dict['Filename']}: {e}")

        print('   Images loaded as arrays')

        try:
            # create additional grayscale array and round it to nearest integer
            red, green, blue = 0, 1, 2
        #     grayscale_array = (0.3 * Red) + (0.59 * Green) + (0.11 * Blue)
        #     grayscale_array = ((0.3 * composite_array[red]) + (0.59 * composite_array[green]) + (0.11 * composite_array[blue]))
            grayscale_array = np.around((0.3 * RGB_array[:, :, red]) + (0.59 * RGB_array[:, :, green]) +  (0.11 * RGB_array[:, :, blue]), 0)
            grayscale_array = grayscale_array.astype('int')
            print("   Grayscale array shape ", grayscale_array.shape)
        #     print(grayscale_array[100:110, 100:110])

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Grayscale calculation error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error calculating grayscale {list_dict['Filename']}: {e}")

        try:
            print("   Composite creation started 2")
            # Reshape train_array to (1, height, width)
            train_array = np.expand_dims(train_array, axis=0)

            # Transpose RGB_array to (3, height, width)
            RGB_array = np.transpose(RGB_array, (2, 0, 1))

            # Reshape grayscale_array to (1, height, width)
            grayscale_array = np.expand_dims(grayscale_array, axis=0)

            # Reshape DMR_array to (1, height, width)
            # DMR_array = np.expand_dims(DMR_array, axis=0)

            # Now all arrays have the same (bands, height, width) format
            # print(f"   Train array shape: {train_array.shape}", end = " ")  # Expected: (1, 8000, 10000)
            # print(f"   RGB array shape: {RGB_array.shape}", end = " ")      # Expected: (3, 8000, 10000)
            # print(f"   Grayscale array shape: {grayscale_array.shape}", end = " ")  # Expected: (1, 8000, 10000)

            # Append along the first axis (bands)
            # composite_array = np.append(train_array, [RGB_array, grayscale_array], axis=0)
            composite_array = np.concatenate((train_array, RGB_array, grayscale_array), axis=0)
            # composite_array = np.concatenate((train_array, RGB_array, grayscale_array, DMR_array), axis=0)
            # Print final shape
            print(f"   Composite array shape: {composite_array.shape}", end = " ")  # Expected: (5, 8000, 10000)
            print("   Composite array created")

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Composite creation error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"  Error compositing {list_dict['Filename']}: {e}")


    # Segmentation
    def segmentation_subroutine(self, composite_array, list_dict, loc, shape):
        status_dict = {}
        try:
            name_elements = list_dict['Filename'].replace(".tif","")
            name_elements = name_elements.split("_")
            # Split raster to calibration polygons
            loc = f"{name_elements[2]}_{name_elements[1]}"

            # print(list_dict)

            # print(loc)
            list_dict['Just_others'] = 'just_others'
            segment = Segmentation(self.db_path, self.model_folder, self.model_number, self.config_json_path)
            class_db_name, class_db_path = segment.run_segmentation(list_dict['Folder'], loc, list_dict['Class_csv'], list_dict['Pixel_number'], composite_array, list_dict['Shift'], list_dict['Modification'], list_dict['Just_others'], shape)

            status_dict['Status'] = "Segmentation done"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)

            error_relay = 0
            return [class_db_name, class_db_path, error_relay]

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Segmentation error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"Error segmenting {list_dict['Filename']}: {e}")

            class_db_name, class_db_path, error_relay = 0, 0, 1
            return [class_db_name, class_db_path, error_relay]




    # Classification
    def classification_subroutine(self, list_dict, class_db_path, model_path):
        status_dict = {}
        try:
            classify = Classification(self.db_path, self.model_folder, self.model_number, self.config_json_path)

            classify.classify_image_db(class_db_path, model_path)

            status_dict['Status'] = "Classification done"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)

            error_relay = 0
            return error_relay

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Classification error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"Error classifying {list_dict['Filename']}: {e}")

            error_relay = 1
            return error_relay


    # Deletion
    def deleting_subroutine(self, list_dict, class_db_path):
        status_dict = {}
        try:

            self.delete_classified(class_db_path)

            status_dict['Status'] = "Deleting done"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)

            error_relay = 0
            return error_relay

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            status_dict['Status'] = "Deleting error"
            self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', list_dict['Segmentation_ID'], status_dict)
            print(f"Error deleting {list_dict['Filename']}: {e}")

            error_relay = 1
            return error_relay





    #%%

    #
    ## Get the number of training segments from traininng db and write them into Segmentation_setting db
    #

    def calculate_statistics(self):

        # dictionary
        segment_dict =  {
            'Segmentation_ID': 'INTEGER PRIMARY KEY',
            'Filename': 'TEXT',
            'Folder': 'TEXT',
            'Train_path': 'TEXT',
            'Class_csv': 'TEXT',
            'Pixel_number': 'INTEGER',
            'Shift': 'INTEGER',
            'Modification': 'TEXT',
            'Just_others': 'TEXT',
            'Status': 'TEXT'
            }

        # open database connection
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function calculate_statistics run for {self.model_folder}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        # print(self.db_path)
        cursor.execute("SELECT * FROM Segmentation_setting WHERE Status IS NOT 'Counting training mosaics done'")
        records = cursor.fetchall()

        print(f"Db {self.db_path} scanned succesfully, {len(records)} rasters found")

        i, tot_time, C = 0, 0, 1
        # iterate through Segmentation_setting db
        for row in records:
            start = time.time()
            settings = {}
            status_dict = {}
            # print(row)

            try:
                list_dict = self.table_manag.retrieve_data_as_dict('Segmentation_setting', 'Segmentation_ID', row[0], segment_dict)
                # print(list_dict)

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Retrieving data as dictionary error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {self.db_path}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            try:
                name_elements = list_dict['Filename'].replace(".tif","")
                name_elements = name_elements.split("_")
                # print(name_elements)
                # Split raster to calibration polygons
                loc = f"{name_elements[2]}_{name_elements[1]}_{list_dict['Shift']}_log"
                # assembly db name
                segments_db_name = f"List_{loc}.db"

                print(f"Analysing db {segments_db_name}")

                # assembly db path
                segments_db_path = os.path.join(self.mosaics_path, segments_db_name)

                # open class for segment table modification
                segment_manag = DB_management(segments_db_path)

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Opening segments db error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {segments_db_name}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            # count classes
            try:
                # print(segments_db_path)
                rill_count = segment_manag.count_lines_not_null('segments', 'Rill')
                settings['Rill'] = rill_count

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Counting Rills error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {segments_db_name}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            try:
                norill_count = segment_manag.count_lines_not_null('segments', 'NoRill')
                settings['NoRill'] = norill_count

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Cunting NoRills error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {segments_db_name}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            try:
                # write into Segementation setting db Rills
                self.table_manag.add_column_to_table('Segmentation_setting', 'Rill_count', 'INTEGER')
                self.table_manag.create_record_from_dict('Segmentation_setting', settings)

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Writing Rills count error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {segments_db_name}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            try:
                # write into Segementation setting db NoRills
                self.table_manag.add_column_to_table('Segmentation_setting', 'NoRill_count', 'INTEGER')
                self.table_manag.create_record_from_dict('Segmentation_setting', settings)

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Writing NoRills count error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {segments_db_name}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            try:
                status_dict['Status'] = "Counting training mosaics done"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Writing mosaics count error"
                self.table_manag.update_record_from_dict('Segmentation_setting', 'Segmentation_ID', row[0], status_dict)
                print(f"Error {status_dict['Status']} {segments_db_name}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error


            # time management and estimated time left
            i, C = self.est_time_left(i, start, tot_time, records, C, trigger = 5)

        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = 'Function calculate_statistics finished'
        self.table_manag.create_record_from_dict('Processing_log', log)

        conn.close()

        print(f"Processing of {i} scenes completed.")

    #%%

    #
    ## Invert colors in training mosaics
    #

    def invert_images(self, category, filetype):
        """
        Invert all images in specified category folder

        :param category: Name of the training class folder.
        :param filetype: Filetype of images.
        """

        image_folder = os.path.join(self.mosaics_path, category)
        folder_content = os.listdir(image_folder)

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function invert_images run for {category}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        # skip any already inverted files
        temp_list, skip_list = [], []
        for filename in folder_content:
            if filename.endswith("." + filetype):

                if f'_inverted.{filetype}' in filename:
                    new_name = filename.replace("_inverted","")
                    skip_list.append(new_name)
                else:
                    temp_list.append(filename)

            else:
                print(f"Filename {filename} has a bad filetype, not {filetype}")

        worklist = []
        for filename in temp_list:
            if filename in skip_list:
                pass
            else:
                worklist.append(filename)

        print(f"Folder_content {len(folder_content)}, skip list {len(skip_list)}, work list {len(worklist)}, ")

        # return

        # go through the worklist and invert the images
        i, tot_time, C = 0, 0, 1
        list_dict = {}

        for filename in worklist:

            # time management
            start = time.time()

            status_dict = {}
            list_dict['Filename'] = filename

            try:
                # Full path to the RGB file
                RGB_path = os.path.join(image_folder, filename)

                # Open the RGB image
                RGB_img = Image.open(RGB_path)
                # print(f'   RGB image {RGB_path} loaded')

                # Convert RGB image into array
                RGB_array = np.array(RGB_img)
                # print(f'   RGB image {RGB_path} converted into array')

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Importing images and converting to array error"
                print(f"Error {status_dict['Status']} {list_dict['Filename']}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            try:
                # Assume image_array is your (3, height, width) RGB image that is actually grayscale
                # Check if it's uint8 (0-255) or float (0-1)
                if RGB_array.dtype == np.uint8:
                    inverted_array = 255 - RGB_array
                else:  # Assume it's float (0-1)
                    inverted_array = 1.0 - RGB_array

                # print(f'   RGB image {RGB_path} inverted')

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Writing NoRills count error"
                print(f"Error {status_dict['Status']} {list_dict['Filename']}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            try:
                new_name = list_dict['Filename'].replace(f".{filetype}","")
                new_name = f"{new_name}_inverted.{filetype}"

                mosaic_save_path = os.path.join(image_folder, new_name)
                im = Image.fromarray(inverted_array)
                im = im.convert('RGB')
                im.save(mosaic_save_path)    # save only training mosaics
                print(f'   {i}/{len(folder_content)} Inverted RGB image saved as {mosaic_save_path}')

            except Exception as e:
                # Handle any errors that occur during processing (so the script doesn't stop)
                status_dict['Status'] = "Saving inverted images"
                print(f"Error {status_dict['Status']} {list_dict['Filename']}: {e}")
                i += 1
                continue  # Move on to the next scene even if there's an error

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            i, C = self.est_time_left(i, start, tot_time, worklist, C, trigger = 20)


        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function invert images {category} finished'
        self.table_manag.create_record_from_dict('Processing_log', log)

    # Paths to your aerial photography scenes (feature class or shapefile) and lines feature class
    # folder = r"\\backup.fsv.cvut.cz\Users\tejklada\DPZ_junior_Tejkl\Models\model_005"
    # invert_images(folder, "Rill", "jpg")
    # invert_images(folder, "NoRill", "jpg")

#%%

    def check_for_cancel(self):
        """
        Checks whether cancel txt is set to yes.
        Returns string cancel, this needs if condition in loop to break loop.
        """
        try:
            cancel_txt = open(self.cancel_txt_path, "r")
            cancel = cancel_txt.read()
            if cancel == "Yes" or cancel == "yes":
                print("    Model canceled due to cancel txt set to Yes")
                return("Cancel")
        except Exception as e:
            logging.error("    Error canceling model", str(e))
            raise


#%%

    def normalize_array_to_uint8(self, array, out_min=0, out_max=255):
        """
        Normalize a NumPy array to the uint8 range [0, 255] for image saving or model input.
        array : np.ndarray 2D NumPy array of real values (e.g., elevation in meters).
        out_min : int, optional. Minimum value of output range. Default is 0.
        out_max : int, optional. Maximum value of output range. Default is 255.
        Returns:
        norm_array : np.ndarray. Normalized array as uint8, with shape same as input, values in [0, 255].
        original_min : float. Minimum value in the original array (useful if you want to un-normalize later).
        original_max : float. Maximum value in the original array.
        """

        original_min = np.amin(array)
        original_max = np.amax(array)
        # print(f"original min: {original_min}, original max {original_max}")

        # Avoid division by zero
        if original_max == original_min:
            norm_array = np.zeros_like(array, dtype=np.uint8)
        else:
            norm_array = ((array - original_min) / (original_max - original_min)) * (out_max - out_min)
            norm_array = np.clip(norm_array, out_min, out_max).astype(np.uint8)

        # print(norm_array, original_min, original_max)
        # print(f"DEBUG: type(original_min): {type(original_min)}")
        print(f'   DEBUG: new min: {np.amin(norm_array)}, new max {np.amax(norm_array)}')
        return norm_array, original_min, original_max

    #%%

    #
    ## Delete classified images
    #

    def delete_classified(self, class_db_path):

        class_table_manag = DB_management(class_db_path)
        records = class_table_manag.iterate_through_table('segments', where_status = 'done is 1')

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function classify_validation_folder run for {self.model_folder}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        i, tot_time, C = 0, 0, 1
        for row in records:

            # time management
            start = time.time()

            # control cancel txt"
            cancel = self.check_for_cancel()
            if cancel == "Cancel":
                break

            mosaic_path = row[0]
            print(f'Mosaic remove path {mosaic_path}')

            # remove image file
            try:
                os.remove(mosaic_path)
            except:
                print(f"Unable to remove {mosaic_path}")

            i, C = self.est_time_left(i, start, tot_time, records, C, trigger = 50)


#%%

#
##
#

    def est_time_left(self, increment, start, tot_time, records, secondary_increment, trigger = 100):
        """
        Manages folder creation when the file count exceeds the threshold.

        Parameters:
        increment (int):
        start (time):
        tot_time (time):
        records (list):
        secondary_increment (int):

        Returns:
        increment (int):
        secondary_increment (int):
        """

        end = time.time()
        tot_time += (end - start)

        if increment == 5:
            print('***')
            print('   Estimated time left: ', end = '')
        if increment % trigger == 0:
            est_time = (len(records) - increment) * (tot_time/ secondary_increment)
            if est_time > 86400:
                print(round(est_time/86400, 2), 'days')
            elif est_time > 3600:
                print(round(est_time/3600, 2), 'hours')
            else:
                print(round(est_time/60, 2), 'mins')
            tot_time, secondary_increment = 0, 1

        # counting mechanism
        increment += 1
        secondary_increment += 1

        print('***')

        return(increment, secondary_increment)
    
    #%%
    
#
##
#

    def get_filename(self, filename, file_extension, separator):
        """
        Manages folder creation when the file count exceeds the threshold.

        Parameters:
        filename (text):
        file_extension (text): in the format of ".tif" or ".jpg"
        separator (text):

        Returns:
        increment (int):
        secondary_increment (int):
        """
        
        if ".tif" in filename:
            filename = filename.replace(file_extension, "")
        filename_split = filename.split(separator)
        loc = filename_split[2]
        year = filename_split[1]

        print(f'   Filename {filename} splitted into location: {loc} and year {year}, file extension {file_extension} removed, splitted by {separator}')
        
        return(loc, year)

    #%%

    #
    ## Run model training
    #

class Classification:

    def __init__(self, db_path, model_folder, model_number, config_json_path):
        """Initialize with the path to the database."""
        self.db_path = db_path
        self.model_folder = model_folder
        self.model_number = model_number
        # self.table_name = table_name

        self.mosaics_path = os.path.join(model_folder, "mosaics")

        # Path to the cancel txt
        self.cancel_txt_path = os.path.join(self.mosaics_path, "Cancel_model.txt")

        self.table_manag = DB_management(self.db_path)
        self.starting_db = Starting_database(self.db_path, self.model_folder, self.model_number, config_json_path)

    #%%

    #
    ## Create new database with columns for later writing classification values
    #

    def create_classification_db(self, validation_db_path, categories):

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function create_classification_db run for {self.model_folder}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        valid_table_manag = DB_management(validation_db_path)

        try:
            # dictionary
            validation_dict =  {
                'Segment_ID': 'INTEGER PRIMARY KEY',
                'Filename': 'TEXT',
                'Folder': 'TEXT'
                }

            # Append new columns with data type REAL
            for item in categories:
                validation_dict[item] = 'REAL'

            valid_table_manag.create_table('Segments_val', validation_dict)

            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = 'Function create_classification_db finished'
            self.table_manag.create_record_from_dict('Processing_log', log)

        except Exception as e:
            logging.error("    Error in create_classification_db: %s", str(e))

            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = 'Function create_classification_db error: {e}'
            self.table_manag.create_record_from_dict('Processing_log', log)

            raise

    #%%

    #
    ## classify_mosaic
    #

    def classify_mosaic(self, mosaic_path, classification_model):
        import tensorflow as tf
        from tensorflow import keras

        """
        Manages folder creation when the file count exceeds the threshold.

        Parameters:
        main_folder (str): Path to the model folder.
        model_path (str): Path to the saved model.
        max_error_count (int): maximum number of errors encountered to trip major error, basically specify maximum error tollerance.

        Returns:
        tuple: Updated others folder name.
        """

        error_count = 0

        try:

            try:

                img = tf.keras.preprocessing.image.load_img(mosaic_path, color_mode='rgb')

                img_array = keras.utils.img_to_array(img)
                # img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                predictions = classification_model.predict(img_array)
                # score = float(keras.ops.sigmoid(predictions[0][0]))
                score = float(tf.nn.sigmoid(predictions[0][0]))
                predictions = [score, 1- score]
        ##        print(predictions)

                return predictions

            except Exception as e:
                error_count += 1
                print(f"Error {e} during analysis of mosaic {mosaic_path}")
                raise


            print(f"Analysis of {mosaic_path} done")

        except Exception as e:
            logging.error("    Error in classify_mosaic: %s", str(e))
            raise

    #%%

    #
    ## Classify rest of segments
    #


    def classify_validation_folder(self, category, model_path, validation_db_path):
        from tensorflow import keras

        # print(f'Db path {self.db_path}')

        try:
            classification_model = keras.models.load_model(model_path)

            # log process in Processing_log
            log = {}
            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = f'Function classify_validation_folder loading model {model_path} succesfull'
            self.table_manag.create_record_from_dict('Processing_log', log)

        except Exception as e:
            logging.error("    Error loading model file {model_path}", str(e))

            # log process in Processing_log
            log = {}
            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = f'Function classify_validation_folder loading model error for {model_path}'
            self.table_manag.create_record_from_dict('Processing_log', log)

            raise

        mosaic_save_path = os.path.join(self.model_folder, category)
        folder_content = os.listdir(mosaic_save_path)

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function classify_validation_folder run for {self.model_folder}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        valid_table_manag = DB_management(validation_db_path)

        new_dict = {}
        new_dict['Folder'] = mosaic_save_path

        i, tot_time, C = 0, 0, 1
        for filename in folder_content:

            # time management
            start = time.time()

            new_dict['Filename'] = filename

            # control cancel txt"
            cancel = self.starting_db.check_for_cancel()
            if cancel == "Cancel":
                break

            mosaic_path = os.path.join(mosaic_save_path, filename)
            print(f'Mosaic save path {mosaic_path}')

            # classify the saved mosaic
            try:
                predictions = self.classify_mosaic(mosaic_path, classification_model)

            except Exception as e:
                logging.error("    Error classify_mosaic", str(e))
                continue

            # treat result
            rounded_predictions = []
            print_predictions = []
            for prediction in predictions:
                num = int(prediction * 100)
                rounded_predictions.append(num)
                symbol = self.treat_format(num, 3, ' ')
                print_predictions.append(symbol)

            percentage = round((i/len(folder_content))*100,2)
            print_clause = f'Image {filename} Rill {print_predictions[1]} NoRill {print_predictions[0]} {percentage} %'
            print(print_clause)

            new_dict['NoRill_val'] = rounded_predictions[0]
            new_dict['Rill_val'] = rounded_predictions[1]

            # write result to db
            try:
                valid_table_manag.create_record_from_dict('Segments_val', new_dict)

            except Exception as e:
                logging.error("    Error write result to db", str(e))
                raise

            # time management and estimated time left
            i, C = self.starting_db.est_time_left(i, start, tot_time, folder_content, C, trigger = 10)


        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = 'Function calculate_statistics finished'
        self.table_manag.create_record_from_dict('Processing_log', log)


#%%

    def treat_format(self, num, places, placeholder):
        symbol = str(num)
        for i in range(places):
            if len(symbol) < places:
                symbol = placeholder + symbol
        return(symbol)

#%%

#
## Classify db
#

    def classify_image_db(self, class_db_path, model_path):
        from tensorflow import keras

        try:
            classification_model = keras.models.load_model(model_path)

            # log process in Processing_log
            log = {}
            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = f'Function classify_validation_folder loading model {model_path} succesfull'
            self.table_manag.create_record_from_dict('Processing_log', log)

        except Exception as e:
            logging.error("    Error loading model file {model_path}", str(e))

            # log process in Processing_log
            log = {}
            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = f'Function classify_validation_folder loading model error for {model_path}'
            self.table_manag.create_record_from_dict('Processing_log', log)

            raise

        class_table_manag = DB_management(class_db_path)
        try:
            print(f"   Classification of {class_db_path} started")
            records = class_table_manag.iterate_through_table('segments', where_status = 'done is 0')

        except Exception as e:
            logging.error("    Error loading model file {model_path}", str(e))

            # log process in Processing_log
            log = {}
            log['Time'] = f'{date.today()} {time.time()}'
            log['Main_log'] = f'Function classify_validation_folder loading model error for {model_path}'
            self.table_manag.create_record_from_dict('Processing_log', log)

        # log process in Processing_log
        log = {}
        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = f'Function classify_validation_folder run for {self.model_folder}'
        self.table_manag.create_record_from_dict('Processing_log', log)

        i, tot_time, C = 0, 0, 1
        for row in records:

            # time management
            start = time.time()


            # control cancel txt"
            cancel = self.starting_db.check_for_cancel()
            if cancel == "Cancel":
                break

            mosaic_path = row[0]
            filename = row[1]
            print(f'Mosaic save path {mosaic_path}')

            update_dict = {}

            # classify the saved mosaic
            try:
                predictions = self.classify_mosaic(mosaic_path, classification_model)

            except Exception as e:
                logging.error("    Error classify_mosaic", str(e))
                continue

            # treat result
            rounded_predictions = []
            print_predictions = []
            for prediction in predictions:
                num = int(prediction * 100)
                rounded_predictions.append(num)
                symbol = self.treat_format(num, 3, ' ')
                print_predictions.append(symbol)

            percentage = round((i/len(records))*100,2)
            print_clause = f'Image {filename} Rill {print_predictions[1]} NoRill {print_predictions[0]} {percentage} %'
            print(print_clause)

            update_dict['NoRill'] = rounded_predictions[0]
            update_dict['Rill'] = rounded_predictions[1]
            update_dict['Done'] = 1

            # write result to db
            try:
                # class_table_manag.create_record_from_dict('segments', new_dict)
                class_table_manag.update_record_from_dict('segments', 'Name', filename, update_dict)

            except Exception as e:
                logging.error("    Error write result to db", str(e))
                raise

            # time management and estimated time left
            i, C = self.starting_db.est_time_left(i, start, tot_time, records, C, trigger = 5)


        log['Time'] = f'{date.today()} {time.time()}'
        log['Main_log'] = 'Function calculate_statistics finished'
        self.table_manag.create_record_from_dict('Processing_log', log)


#%%

#
## Rename files because you forgot to change the new name
#

# folder = r"S:\Projects\TACR-SIGMA-EROZE-ANN\DMR4g"
# folder = r"S:\Projects\TACR-SIGMA-EROZE-ANN\DMR5g"
# folder = r"S:\Projects\TACR-SIGMA-EROZE-ANN\Flow_acc_DMR4g"
# folder = r"S:\Projects\TACR-SIGMA-EROZE-ANN\Flow_acc_DMR5g"
# folder = r"\\data.fsv.cvut.cz\Projects\TACR-SIGMA-EROZE-ANN\Flow_acc_DMR4g"

def rename_files(folder, new_name):
    folder_content = os.listdir(folder)

    i = 0
    for filename in folder_content:
        break
        try:
            name_elements = filename.split("_")

            old_name = filename
            new_name = f"Flow_acc_DMR4g_{name_elements[1]}"

            if filename == "Cancel_model.txt":
                continue

            old_path = os.path.join(folder, old_name)
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)

            print(f"{i}/{len(folder_content)}; File: {old_name} renamed to: {new_name}")
            i += 1

        except Exception as e:
            # Handle any errors that occur during processing (so the script doesn't stop)
            print(f"   Error {old_name} error: {e}")
            i += 1
            continue  # Move on to the next scene even if there's an error


#%%

class Validation():
    def __init__(self, db_path, model_folder, model_number, config_json_path):
        """Initialize with the path to the database."""
        self.db_path = db_path
        self.model_folder = model_folder
        self.model_number = model_number
        # self.table_name = table_name

        self.mosaics_path = os.path.join(self.model_folder, "mosaics")

        # Path to the cancel txt
        self.cancel_txt_path = os.path.join(self.mosaics_path, "Cancel_model.txt")

        self.table_manag = DB_management(self.db_path)
        self.starting_db = Starting_database(self.db_path, self.model_folder, self.model_number, config_json_path)
        self.debug_manag = Debug_Manager(self.model_folder, self.db_path, config_json_path)



#%%

#
##
#

    def validate_db(self):

        table_manag = DB_management(self.db_path)

        records = table_manag.iterate_through_table('Segments_val')

        Rill_Rill, Rill_NoRill, NoRill_Rill, NoRill_NoRill, Rill_Unkn, NoRill_Unkn, Other, val_Rill, val_NoRill = 0, 0, 0, 0, 0, 0, 0, 0, 0

        i, tot_time, C = 0, 0, 1
        for row in records:
            start = time.time()

            # control cancel txt"
            cancel = self.starting_db.check_for_cancel()
            if cancel == "Cancel":
                break

            print(row)

            line_end = row[1].split("_")
            print(line_end[4])

            if line_end[4] == '7000' and row[3] > 80:
                val_Rill += 1
                Rill_Rill += 1
                print("True Rill")

            elif line_end[4] == '7000' and row[4] > 80:
                val_Rill += 1
                Rill_NoRill += 1
                print("False NoRill")

            elif line_end[4] == '9000' and row[3] > 80:
                val_NoRill += 1
                NoRill_Rill += 1
                print("False Rill")

            elif line_end[4] == '9000' and row[4] > 80:
                val_NoRill += 1
                NoRill_NoRill += 1
                print("True NoRill")

            elif line_end[4] == '7000' and row[3] < 80:
                val_Rill += 1
                Rill_Unkn += 1
                print("Unknown Rill")

            elif line_end[4] == '9000' and row[4] < 80:
                val_NoRill += 1
                NoRill_Unkn += 1
                print("Unknown NoRill")

            else:
                Other += 1
                print("Other happened")


            # if i == 50:
            #     break

            # time management and estimated time left
            i, C = self.starting_db.est_time_left(i, start, tot_time, records, C, trigger = 50)

        print(Rill_Rill, Rill_NoRill, NoRill_Rill, NoRill_NoRill, Rill_Unkn, NoRill_Unkn, Other, val_Rill, val_NoRill)

        #
        ## write stats
        #

        table_columns =  {
            "Line_ID": "INTEGER PRIMARY KEY",
            "Stat": "TEXT",
            "Value": "INTEGER"}

        try:
            self.table_manag.drop_table('validation')
            self.table_manag.create_table('validation', table_columns)
        except:
            self.table_manag.create_table('validation', table_columns)

        new_dict = {}
        new_dict["Stat"] = 'Rill_Rill'
        new_dict["Value"] = Rill_Rill
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'Rill_NoRill'
        new_dict["Value"] = Rill_NoRill
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'NoRill_Rill'
        new_dict["Value"] = NoRill_Rill
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'NoRill_NoRill'
        new_dict["Value"] = NoRill_NoRill
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'Rill_Unkn'
        new_dict["Value"] = Rill_Unkn
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'NoRill_Unkn'
        new_dict["Value"] = NoRill_Unkn
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'Other'
        new_dict["Value"] = Other
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'Total_validation_Rills'
        new_dict["Value"] = val_Rill
        self.table_manag.create_record_from_dict('validation', new_dict)

        new_dict = {}
        new_dict["Stat"] = 'Total_validation_NoRills'
        new_dict["Value"] = val_NoRill
        self.table_manag.create_record_from_dict('validation', new_dict)

        #
        ## write validation matrix
        #

        table_columns =  {
            "Line_ID": "INTEGER PRIMARY KEY",
            "Validation": "TEXT",
            "Class_Rill": "INTEGER",
            "Class_NoRill": "INTEGER",
            "Class_NotSure": "INTEGER"}

        try:
            self.table_manag.drop_table('validation_matrix')
            self.table_manag.create_table('validation_matrix', table_columns)
        except:
            self.table_manag.create_table('validation_matrix', table_columns)

        new_dict = {}
        new_dict["Validation"] = 'Rill'
        new_dict["Class_Rill"] = Rill_Rill / val_Rill * 100
        new_dict["Class_NoRill"] = Rill_NoRill / val_Rill * 100
        new_dict["Class_NotSure"] = Rill_Unkn / val_Rill * 100
        self.table_manag.create_record_from_dict('validation_matrix', new_dict)
        print(new_dict)

        new_dict = {}
        new_dict["Validation"] = 'NoRill'
        new_dict["Class_Rill"] = NoRill_Rill / val_NoRill * 100
        new_dict["Class_NoRill"] = NoRill_NoRill / val_NoRill * 100
        new_dict["Class_NotSure"] = NoRill_Unkn / val_NoRill * 100
        self.table_manag.create_record_from_dict('validation_matrix', new_dict)
        print(new_dict)


#%%

class CMD_run():
    def __init__(self, root, config_json_path):
        """Initialize with the path to the database."""
        self.root = root
        # self.model_folder = model_folder
        self.config_json_path = config_json_path
        # self.table_name = table_name

        # self.mosaics_path = os.path.join(self.model_folder, "mosaics")

        # self.db_path = os.path.join(root, 'scene_db.db')
        self.db_path = os.path.join(root, 'model_zajmove_obdelniky.db')

        # Path to the cancel txt
        # self.cancel_txt_path = os.path.join(self.mosaics_path, "Cancel_model.txt")

        self.table_manag = DB_management(self.db_path)

        # load json paths from config file
        self.paths_dict = self.table_manag.load_table_name_and_columns('json_paths', self.config_json_path)

        # load data from config file
        self.data_dict = self.table_manag.load_table_name_and_columns('data', self.config_json_path)
        # print(self.data_dict)
        self.starting_db = Starting_database(self.db_path, root, self.data_dict['model_number'], config_json_path)

        # load table templates
        self.db_template_path = os.path.join(self.root, self.paths_dict['tables'])
        
#%%

#
## Create new db for classified scene
#

    def create_fresh_dbs(self):
        """
        Creates new db and fills it with empty lists

        Returns
        -------
        None.
        """
        table_names = {'metadata': 'metadata', 'Segmentation_setting': 'segment_dict','Processing_log': 'processing_columns'}

        # iterate through table_names list and create particular tables
        for key, value in table_names.items():
            # create table
            # print(key, value)
            try:
                db_template = self.table_manag.load_table_name_and_columns(value, self.db_template_path)
                self.table_manag.create_table(key, db_template)
                self.debug_manag.debug("   Function create_fresh_db run succesfully", statement_level=1)

            except Exception as e:
                print(f"Error function create_fresh_db, table name {key}, error {e}")


#%%

#
##
#

    def fill_table_with_metadata(self):
        """
        Creates new db and fills it with empty lists

        Returns
        -------
        None.
        """
        import rasterio

        data_dict = self.table_manag.load_table_name_and_columns('data', self.config_json_path)

        metadata_dict = {}
        metadata_dict['Filename'] = data_dict['scene']
        # metadata_dict['List_name'] = data_dict['scene']
        metadata_dict['Pixel_number'] = data_dict['Pixel_number']
        metadata_dict['Folder'] = self.root
        metadata_dict['Status'] = 'Segmentation setting created'
        # metadata_dict['Train_status'] = 'No training lines'

        # open image and get metadata
        # image_path = os.path.join(self.root, data_dict['scene'])
        image_path = os.path.join(self.root, 'scenes', data_dict['scene'])
        with rasterio.open(image_path) as scene:
            metadata_dict['Width'] = scene.width
            metadata_dict['Height'] = scene.height
            metadata_dict['Spatial_Reference'] = scene.crs
            # metadata_dict['Spatial_Reference'] = 'S-JTSK_Krovak_East_North'
            # print("Width (px):", scene.width)
            # print("Height (px):", scene.height)
            # print("Coordinate system:", scene.crs)
            # print("Transform (affine):", scene.transform)

            # Calculate bounds (corner coordinates)
            bounds = scene.bounds
            metadata_dict['X_Min'] = bounds.bottom
            metadata_dict['Y_Min'] = bounds.left
            metadata_dict['X_Max'] = bounds.top
            metadata_dict['Y_Max'] = bounds.right
            # print("Bounds:", bounds)
            # print("Top-left:", (bounds.left, bounds.top))
            # print("Bottom-right:", (bounds.right, bounds.bottom))

            # Pixel size (resolution)
            pixel_size_x = scene.transform[0]
            # pixel_size_y = -scene.transform[4]
            metadata_dict['Cell_size'] = pixel_size_x
            # print("Pixel size (X):", pixel_size_x)
            # print("Pixel size (Y):", pixel_size_y)

        try:
            self.debug_manag.debug(f"   Metadata dict {metadata_dict}", statement_level=1)
            self.table_manag.create_record_from_dict('metadata', metadata_dict)
            # self.table_manag.create_record_from_dict('Lists', metadata_dict)

            print("   Function create_fresh_db run succesfully")

        except Exception as e:
            print(f"Error function fill_table_with_metadata, error {e}")


#%%

#
## fill Segmentation_setting
#

    def fill_segmentation_setting(self, shift):
        """
        Creates new db and fills it with empty lists

        Inputs
        -------
        :param shift:

        Returns
        -------
        None.
        """

        data_dict = self.table_manag.load_table_name_and_columns('data', self.config_json_path)
        csv_paths_dict = self.table_manag.load_table_name_and_columns('csv_paths', self.config_json_path)
        # set db
        # segment_dict = self.table_manag.load_table_name_and_columns('segment', os.path.join(self.root, self.paths_dict['tables']))
        segment_dict = {}
        segment_dict['Filename'] = data_dict['scene']
        segment_dict['Folder'] = self.root
        segment_dict['Class_csv'] = os.path.join(self.root, csv_paths_dict['class_setting'])
        segment_dict['Pixel_number'] = data_dict['Pixel_number']
        segment_dict['Shift'] = shift
        segment_dict['Modification'] = 'none'
        segment_dict['Just_others'] = 'just_others'
        segment_dict['Status'] = 'Segmentation setting created'

        try:
            self.debug_manag.debug(f"   Metadata dict {segment_dict}", statement_level=1)
            self.table_manag.create_record_from_dict('Segmentation_setting', segment_dict)

            print("   Function create_fresh_db run succesfully")
            return data_dict['scene']

        except Exception as e:
            print(f"Error function fill_table_with_metadata, error {e}")


#%%

#
## segment and classify
#

    def segment_and_classify(self):
        """
        Creates new db and fills it with empty lists

        Returns
        -------
        None.
        """

        try:
            model_path = os.path.join(self.root, self.data_dict['model'])

            # print('   ', scene_path)
            # print('   ', self.data_dict['shape'])
            # print('   ', model_path)
            # print('   ', self.root)

            self.starting_db.segment_and_classify_scenes(self.root, self.data_dict['shape'], model_path, inverse = 0)

        except Exception as e:
            print(f"Error function segment_and_classify, error {e}")


#%%

#
## Creates raster
#

    def create_classified_array(self, Segmentation_ID, column_name):
        """
        Creates new db and fills it with empty lists

        Inputs
        -------
        :param Segmentation_ID:
        :param column_name:

        Returns
        -------
        None.
        """

        try:
            metadata_dict = self.table_manag.load_table_name_and_columns('metadata', self.db_template_path)
            metadata = self.table_manag.retrieve_data_as_dict('metadata', 'List_ID', 1, metadata_dict)

            list_dict = self.table_manag.load_table_name_and_columns('segment_dict', self.db_template_path)
            list_setting = self.table_manag.retrieve_data_as_dict('segment_dict', 'Segmentation_ID', Segmentation_ID, list_dict)

            print(f"   Metadata: width {metadata['width']}, height {metadata['height']}, pixel_number {metadata['pixel_number']}, shift {list_setting['shift']}, cellSize {metadata['Cell_size']}")
        #     return()
            # create empty raster
            resultArray = np.zeros((metadata['height'], metadata['width']), dtype=int)

            # recognize shift
            s, t = 0, 0
            if list_setting['shift'] == 1:
                s = int(metadata['pixel_number']/2)
                t = int(metadata['pixel_number']/2)
            elif list_setting['shift'] == 2:
                t = int(metadata['pixel_number']/2)
            elif list_setting['shift'] == 3:
                s = int(metadata['pixel_number']/2)
            else:
                pass

            ## create name of the classified db
            # Get location
            loc, year = self.starting_db.get_filename(list_dict['Filename'], self.filetype, self.separator)

            # Get class_db_name
            class_db_name = f"List_{loc}_{year}_{list_setting['Shift']}_log.db"
            class_db_path = os.path.join(self.root, 'mosaics', class_db_name)

            # Iterate through all records in the DB
            conn = sqlite3.connect(class_db_path)
            cursor = conn.cursor()

            sql_query = f"SELECT * FROM segments WHERE {column_name} != 0"

            cursor.execute(sql_query)
            records = cursor.fetchall()

            print(f"   DB {class_db_path} scanned succesfully, {len(records)} lines found")
            i, tot_time, C = 0, 0, 1

            segment_dict = self.table_manag.load_table_name_and_columns('segment_columns', self.db_template_path)
            segment_setting = self.table_manag.retrieve_data_as_dict('segments', 'List_ID', 1, segment_dict)

            for row in records:
                try:
                    start = time.time()
                    # print(row)

                    # control cancel txt"
                    cancel = self.starting_db.check_for_cancel()
                    if cancel == "Cancel":
                        break

                    segment = self.table_manag.retrieve_data_as_dict('segments', 'Name', row[1], segment_setting)

                    X = int(segment['X_col'])
                    Y = int(segment['Y_row'])
                    value = segment[column_name]
                    # print(X, Y, value)

                    k = Y * metadata['pixel_number'] + s
                    l = X * metadata['pixel_number'] + t
                    resultArray[k:k+metadata['pixel_number'], l:l+metadata['pixel_number']] = int(value)

                    # if i%4000 == 0:
                    #     print(resultArray[k:k + 15, l:l + 15])
                        # print(resultArray[int(height/2) : int(height/2) + 10, int(width/2) : int(width/2) + 10])

                    # time management and estimated time left
                    i, C = self.starting_db.est_time_left(i, start, tot_time, records, C, trigger = 2000)

                except Exception as e:
                    print(f"Error function segment_and_classify, error {e}")


            # print(f'   Array stats: min {resultArray.min()}, max {resultArray.max()}, shape {resultArray.shape}')
            # unique_vals, counts = numpy.unique(resultArray, return_counts=True)
            # print(f'   Unique values:", {list(zip(unique_vals.tolist(), counts.tolist()))}')
            print('   Raster for ', column_name, ' done')

            return(resultArray)

        except Exception as e:
            print(f"Error function segment_and_classify, error {e}")


#%%

#
## Average it
#

    def average_classified_arrays(self, filename, column_name):
        """
        Creates new db and fills it with empty lists

        Inputs
        -------
        :param filename:
        :param column_name:

        Returns
        -------
        None.
        """

        print("   Tool average_classified_arrays started")
        try:
            records = self.table_manag.iterate_through_table('segment_dict', where_status = f"Filename IS {filename}")
            print("   Total records found {len(records)}")

            array_counter = 0
            for row in records:

                if array_counter == 0:
                    newArray = self.create_classified_array(row[0], column_name)

                elif array_counter == 1:
                    resultArray = self.create_classified_array(row[0], column_name)
                    composite_array = np.concatenate((newArray, resultArray), axis=0)

                else:
                    resultArray = self.create_classified_array(row[0], column_name)
                    composite_array = np.concatenate((composite_array, resultArray), axis=0)

                print("   create_classified_array with i = {i}, done")
                array_counter += 1

        except Exception as e:
            print(f"Error function average_classified_arrays, creating classified arrays, error {e}")

        try:
            # composite: (bands, height, width)
            averaged_array = np.mean(composite_array, axis=0)   # average per pixel
            print("   Mean array shape:", averaged_array.shape)
            print("   Min/Max:", averaged_array.min(), averaged_array.max())

            return(averaged_array)

        except Exception as e:
            print(f"Error function average_classified_arrays, averaging arrays, error {e}")


#%%

#
## Filter it
#

    def filter_array(self, averaged_array):
        """
        Creates new db and fills it with empty lists

        Inputs
        -------
        :param filename:
        :param averaged_array:

        Returns
        -------
        None.
        """

        print("   Tool average_classified_arrays started")

        ## array dimensions
        dimensions = averaged_array.shape

        print(f"   Preparation of compo array {averaged_array} is done")
        print(f"   Dimensions: {dimensions}")


        height, width = dimensions[0], dimensions[1]

        # empty result arrays

        result_array = np.zeros((height, width))
        executed_array = np.zeros((height, width))
        process_array = np.zeros((height, width))

        # statistical data
        statistics = {}

        # seting the basic parameters
        max_x = width-1
        max_y = height-1
        # max_x = 1000
        # max_y = 1000
        ra = width * height

        self.debug_manag.debug(f"   {max_x}, {max_y}", statement_level=1)

        ## vse nad 50 beru pixel
        try:
            side = 0
            i,j = 0, 0
            a = 0
            c = 0

            for k in range(ra):
                self.count_event(statistics, 'prvni_strom')
                c += 1

                if averaged_array[j,i] >= 50 and averaged_array[j,i] <= 100:
                    result_array[j,i] = 1
                    executed_array[j,i] = 1
                    self.count_event(statistics, 'vse_nad_50')

                else:
                    pass

                if i == max_x:
                    i = side
                    j += 1
                    a += 1
                else:
                    i += 1

                if j == max_y:
                    break

        except Exception as e:
            print(f"Error function filter_array, vse nad 50 beru pixel, error {e}")

        # filtrace osamelych pixelu
        try:
            side = 0
            i,j = 0, 0
            a = 0
            c = 0

            for k in range(ra):
                self.count_event(statistics, 'druhy_strom')
                c += 1

                if averaged_array[j,i] >= 50  and averaged_array[j,i] <= 100:
                    self.count_event(statistics, 'mezi_50_100')

                    try:
                        leva_horni_sum = result_array[j-1, i-1] + result_array[j-1, i] + result_array[j-1, i+1] + result_array[j, i-1] + result_array[j+1, i-1]
                        if leva_horni_sum == 0:
                            self.count_event(statistics, 'leva_horni')

                        prava_horni_sum = result_array[j-1, i-1] + result_array[j-1, i] + result_array[j-1, i+1] + result_array[j, i+1] + result_array[j+1, i+1]
                        if leva_horni_sum == 0:
                            self.count_event(statistics, 'prava_horni')

                        leva_dolni_sum = result_array[j-1, i-1] + result_array[j, i-1] + result_array[j+1, i-1] + result_array[j+1, i] + result_array[j+1, i+1]
                        if leva_horni_sum == 0:
                            self.count_event(statistics, 'leva_dolni')

                        prava_dolni_sum = result_array[j-1, i+1] + result_array[j, i+1] + result_array[j+1, i-1] + result_array[j+1, i] + result_array[j+1, i+1]
                        if leva_horni_sum == 0:
                            self.count_event(statistics, 'prava_dolni')

                        if leva_horni_sum == 0 or prava_horni_sum == 0 or leva_dolni_sum == 0 or prava_dolni_sum == 0:
                            result_array[j,i] = 0
                            executed_array[j,i] = 1
                            self.count_event(statistics, 'osamely_pixel')
                        else:
                            result_array[j,i] = 1
                            executed_array[j,i] = 1
                            self.count_event(statistics, 'neosamely_pixel')

                    except:
                        process_array[j,i] = 1
                        self.count_event(statistics, 'except_osamely')
                        pass

                else:
                    self.count_event(statistics, 'mezi_0_50')

                if i == max_x:
                    i = side
                    j += 1
                    a += 1
                else:
                    i += 1

                if j == max_y:
                    break

        except Exception as e:
            print(f"Error function filter_array, filtrace osamelych pixelu, error {e}")

        ## vse nad 75% i s okolim
        try:
            side = 0
            i,j = 0, 0
            a = 0
            c = 0

            for k in range(ra):
                self.count_event(statistics, 'treti_strom')
                c += 1

                if averaged_array[j,i] >= 75  and averaged_array[j,i] <= 100:
                    try:
                        result_array[j-1:j+2, i-1:i+2] = 1
                        executed_array[j-1:j+2, i-1:i+2] = 1
                        self.count_event(statistics, 'mezi_75_100')

                    except:
                        process_array[j,i] = 1
                        self.count_event(statistics, 'except_75_100')
                else:
                    self.count_event(statistics, 'mezi_0_75')

                if i == max_x:
                    i = side
                    j += 1
                    a += 1
                else:
                    i += 1

                if j == max_y:
                    break

        except Exception as e:
            print(f"Error function filter_array, filtrace osamelych pixelu, error {e}")

        print("   Statistics:")
        print(f"   {statistics}")

        result_array = result_array.astype(int)

        print("   Analysis of array filter_array is done")

        return(result_array)


#%%

    def count_event(self, counter_dict, key):
        """Increment the count for the given key in the given dictionary."""
        counter_dict[key] = counter_dict.get(key, 0) + 1

#%%

#
## Judge it
#

    def judge_filtered_matrix(self, input_NoRill_array, input_Rill_array):
        """
        Creates new db and fills it with empty lists

        Inputs
        -------
        :param input_NoRill_array:
        :param input_Rill_array:

        Returns
        -------
        None.
        """

        print("   Tool judge_filtered_matrix started")
        ## array dimensions
        dimensions = input_NoRill_array.shape
        height, width = dimensions[0], dimensions[1]

        # self.debug_manag.debug(f"   Shape {input_NoRill_array.shape}, {input_Sheet_array.shape}, {input_Rill_array.shape}", statement_level=1)
        self.debug_manag.debug(f"   Shape {input_NoRill_array.shape}, {input_Rill_array.shape}", statement_level=1)

        result_array = np.zeros((height, width))
        executed_array = np.zeros((height, width))

        # statistical data
        statistics = {}

        # seting the basic parameters
        max_x = width-1
        max_y = height-1
        ra = width * height

    #     NoRill, Sheet, Rill = 0, 1, 2
        NoRill, Rill = 0, 1

        # zatridovaci strom

        side = 0
        i,j = 0, 0
        a = 0
        c = 0
        percento = 0
        switch = 0
        C = 0
        print(max_x, max_y)
        for k in range(ra):
            self.count_event(statistics, 'hlavni_strom')
            c += 1

            if input_NoRill_array[j,i] == 1:
                result_array[j,i] = 0
                executed_array[j,i] = 1
                self.count_event(statistics, 'NoRill_1')
    #         elif input_Sheet_array[j,i] == 1:
    #             result_array[j,i] = 1
    #             executed_array[j,i] = 1
    #             self.count_event(statistics, 'Sheet_1')
            elif input_Rill_array[j,i] == 1:
                result_array[j,i] = 2
                executed_array[j,i] = 1
                self.count_event(statistics, 'Rill_1')
            else:
                result_array[j,i] = 4
                executed_array[j,i] = 1
                self.count_event(statistics, 'Else')

            if i == max_x:
                i = side
                j += 1
                a += 1
            else:
                i += 1

            if j == max_y:
                break

            percento_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]

            if round(c/ra *100, 0) in percento_list and switch == 0:
                percento += 10
                switch = 1
                print(str(percento), "% ,", end = ' ')
            elif round(c/ra *100, 0) not in percento_list:
                switch = 0
            else:
                pass

        self.debug_manag.debug(f"   Statistics {statistics}", statement_level=1)

        result_array = result_array.astype(int)


        print("   Judging is done")

#%%

#
## Save it
#

    def save_as_image(self, input_array):
        """
        Creates new db and fills it with empty lists

        Inputs
        -------
        :param input_NoRill_array:
        :param input_Rill_array:

        Returns
        -------
        None.
        """

        print("   Tool save_as_image started")

        import rasterio

        try:

            metadata_dict = self.table_manag.load_table_name_and_columns('metadata', self.db_template_path)
            metadata = self.table_manag.retrieve_data_as_dict('metadata', 'List_ID', 1, metadata_dict)

            # Get location
            loc, year = self.starting_db.get_filename(metadata['Filename'], self.filetype, self.separator)

            result_name = f"{loc}_{year}_classified.tif"
            result_path = os.path.join(self.root, result_name)

            profile = {
                'driver': 'GTiff',
                'dtype': 'int',
                'count': 1,
                'height': metadata['Height'],
                'width': metadata['Width'],
                'crs': metadata['Spatial_reference'],  # your coordinate system
                'transform': ''
            }

            with rasterio.open(result_path, 'w', **profile) as dst:
                dst.write(input_array, 1)

        except Exception as e:
            print(f"Error function save_as_image, error {e}")


#%%











