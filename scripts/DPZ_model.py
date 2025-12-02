# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:08:43 2025

@author: Adam Tejkl
"""

import os
import sqlite3
import logging
from datetime import date
import time
import json


## paths

# model_path = r"Z:\DPZ_junior_Tejkl\Models"
# training_rasters_path = r"Z:\DPZ_junior_Tejkl\Models\training_rasters"
# RGB_rasters_path = r'S:\Projects\TACR-SIGMA-EROZE-ANN\cuzk_ortofoto\data_epsg5514'
# flow_acc_rasters_path = r"Z:\DPZ_junior_Tejkl\Models\model_003"


# model_number = '003'
# folder_path = os.path.join(model_path, 'model_' + model_number)

# # get database path
# db_path = os.path.join(folder_path, 'model_' + model_number + '.db')

#%%

class DB_management:

    def __init__(self, db_path):
        """Initialize with the path to the database."""
        self.db_path = db_path
        # self.table_name = table_name


    def create_table(self, table_name, table_columns):
        """
        Create the specified table in the SQLite database using a dictionary of updated values.

        :param db_path: Path to the SQLite database file.
        :param table_name: Name of the table where the record will be updated.
        :param table_columns: A dictionary containing the columns names and data types.
        """
        # connect to database and create table
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            # build column string
            columns = ", ".join([f"{col_name} {col_type}" for col_name, col_type in table_columns.items()])

            # create table
            c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")

            # commit changes
            conn.commit()

        # print confirmation message
        print(f"Table '{table_name}' created successfully in database '{self.db_path}'")


    def read_from_table(self, table_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {table_name}")

        print(f'Reading from table: {table_name}')
        for row in cursor.fetchall():
            print(row)

        cursor.close()
        conn.close()


    def create_record_from_dict(self, table_name, new_dict):
        """
        Update a record in the specified table in the SQLite database using a dictionary of updated values.

        :param db_path: Path to the SQLite database file.
        :param table_name: Name of the table where the record will be updated.
        :param unit_ID: The unit_ID of the row to update (used as the condition).
        :param update_dict: A dictionary containing the columns to update and their new values.
        :return: True if the update was successful, False otherwise.
        """
        # Generate column names (keys) and placeholders for the values
        columns = ', '.join(new_dict.keys())  # Keys will be column names
        placeholders = ', '.join('?' for _ in new_dict)  # Placeholder for each value

        # Prepare the SQL query for inserting data
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Execute the insert query with the values from the dictionary
            cursor.execute(sql, tuple(new_dict.values()))

            # Commit the transaction and close the connection
            conn.commit()

            print(f"Record in table '{table_name}' created successfully.")
            return True

        except sqlite3.Error as e:
            print(f"An creating error in table '{table_name}' occurred: {e}")
            return False

        finally:
            # Close the connection
            conn.close()


    def retrieve_data_as_dict(self, table_name, ID_column, unit_ID, column_dict_template):
        """
        Retrieve data from a specified table in SQLite and return it as a populated dictionary based on the provided template.

        :param table_name: Name of the table to retrieve data from.
        :ID_column: name of the column where to look for unit_ID
        :param unit_ID: The unit_ID of the row to fetch.
        :param column_dict_template: A dictionary that defines the structure of the required data.
                                     The dictionary keys will be used as column names.
        :return: A dictionary populated with values from the database for the specified row.
        """
        # Create a copy of the template dictionary to use as our result dictionary
        result_dict = column_dict_template.copy()

        # Generate the column list from the keys of the dictionary template
        columns = ', '.join(column_dict_template.keys())

        # SQL query to fetch the data for the given unit_ID
        query = f"SELECT {columns} FROM {table_name} WHERE {ID_column} = ?"

        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Execute the query with the given unit_ID
            cursor.execute(query, (unit_ID,))
            row = cursor.fetchone()

            if row is None:
                print(f"No data found for {ID_column} {unit_ID} in table {table_name}.")
                return None

            # Populate the result dictionary with the data from the row
            for idx, key in enumerate(column_dict_template.keys()):
                result_dict[key] = row[idx]

        except sqlite3.Error as e:
            print(f"An retrieving error occurred: {e}")
            return None

        finally:
            # Close the database connection
            conn.close()

        return result_dict


    def update_record_from_dict(self, table_name, ID_column, unit_ID, update_dict):
        """
        Update a record in the specified table in the SQLite database using a dictionary of updated values.

        :param table_name: Name of the table where the record will be updated.
        :param unit_ID: The unit_ID of the row to update (used as the condition).
        :param update_dict: A dictionary containing the columns to update and their new values.
        :return: True if the update was successful, False otherwise.
        """
        # Generate the SET clause dynamically based on the dictionary keys
        set_clause = ', '.join([f"{key} = ?" for key in update_dict.keys()])
        values = list(update_dict.values())  # Get the values to be updated
        values.append(unit_ID)  # Append the unit_ID for the WHERE clause at the end

        # SQL query to update the record based on unit_ID
        query = f"UPDATE {table_name} SET {set_clause} WHERE {ID_column} = ?"

        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Execute the update query with the values
            cursor.execute(query, values)

            # Commit the changes to the database
            conn.commit()

            print(f"Record in table '{table_name}' updated successfully.")
            return True

        except sqlite3.Error as e:
            print(f"An updating error occurred: {e}")
            return False

        finally:
            # Close the connection
            conn.close()

    # read_from_table(db_path, 'machine')
    # read_from_table(db_path, 'machine_tracking')


    def drop_table(self, table_name):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f'DROP TABLE {table_name}')
        print(f"Table '{table_name}' dropped")

        conn.commit()
        conn.close()

    # drop_table(db_path, 'machine')
    # drop_table(db_path, 'machine_tracking')


    def print_column_names(self, table_name):
        conn = sqlite3.connect(self.db_path)

        # Retrieve the column names for the specified table
        cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 0")
        column_names = [description[0] for description in cursor.description]

        # Print out the column names
        print(f'Printing columns from table: {table_name}')
        print(column_names)
        return()


    def insert_into_table(self, table_name, new_record_line, columns_list):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        columns_clause = ', '.join([f'{column}' for column in columns_list])
        value_clause = ', '.join(['?' for column in columns_list])
        # print(columns_clause)
        # print(value_clause)

        sql = f' INSERT INTO {table_name} ({columns_clause}) VALUES ({value_clause})'

        cursor.execute(sql, new_record_line)

        conn.commit()
        cursor.close()
        conn.close()


    def update_unit_in_table(self, table_name, unit_ID, update_list, columns_list):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create the columns clause for the UPDATE statement
        columns_clause = ', '.join(f"{column} = ?" for column in columns_list)

        # Construct the full SQL UPDATE statement
        sql = f"UPDATE {table_name} SET {columns_clause} WHERE unit_ID = ?"

        update_list.append(unit_ID)

        cursor.execute(sql, update_list)

        conn.commit()
        cursor.close()
        conn.close()

        return()
    
    
    def delete_record(self, table_name, ID_column, unit_ID):
        # Connect to your database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
                
        # Construct the full SQL UPDATE statement
        sql = f"DELETE FROM {table_name} WHERE {ID_column} = ?"
        
        # print(sql, unit_ID)
        print(f'Deleting record {unit_ID}, from {table_name}')
        cursor.execute(sql, (unit_ID,))
                
        conn.commit()
        cursor.close()
        conn.close()


    def refresh_logs(self, logs_list):
        for item in logs_list:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete all records from the table
            cursor.execute(f'DELETE FROM {item}')
            print(f'Tables: {logs_list} refreshed')

            # Commit the changes and close the connection
            conn.commit()
            cursor.close()
            conn.close()


    def drop_tables(self, table_list):
        for item in table_list:
            # Drop all tables in table_list
            self.drop_table(item)


    def get_db_content(self):
        # Connect to the SQLite database (replace 'database.db' with your database file)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the names of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Print and append the names of tables
        table_list = []
        i = 0
        for table in tables:
            if i == 0:
                i += 1
                pass
            else:
                print(f'Tables of database: {self.db_path}')
                print(table[0])
                table_list.append(table[0])

        # Close the connection
        cursor.close()
        conn.close()

        return (table_list)

    # print unit values


    def print_table_row(self, table_name, unit_ID):
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor() # Create a cursor object to interact with the database

        # Check if the table exists
        cursor.execute("PRAGMA table_info({})".format(table_name))
        table_info = cursor.fetchall()

        if not table_info:
            print(f"Table '{table_name}' does not exist in the database.")
            return

        # Construct a SELECT query to fetch the row for the specified unit_ID
        select_query = f"SELECT * FROM {table_name} WHERE unit_ID = ?"

        # Execute the query with the specified unit_ID
        cursor.execute(select_query, (unit_ID,))
        row = cursor.fetchone()

        if row:
            # Print column names and values
            for i, column_info in enumerate(table_info):
                column_name = column_info[1]
                column_value = row[i]
                print(f"{column_name}: {column_value}")
        else:
            print(f"No data found for unit_ID {unit_ID} in table '{table_name}'.")

        # Close the cursor and the database connection
        cursor.close()
        conn.close()


    def get_table_length(self, table_name):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = f"SELECT COUNT(*) FROM {table_name};"
                cursor.execute(query)
                result = cursor.fetchone()
                return result[0]  # The count is the first item in the result tuple
        except sqlite3.Error as e:
            print(f"An error counting table length occurred: {e}")
            return None


    def add_column_to_table(self, table_name, column_name, column_type):
        """
        Add a new column to an existing SQLite table.

        :param table_name: Name of the table to modify.
        :param column_name: Name of the new column to add.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};"
        print(alter_query)
        cursor.execute(alter_query)

        conn.commit()
        conn.close()
        print(f"Added column '{column_name}' to table '{table_name}'.")


    def count_lines_not_null(self, table_name, column_name):
        """
        Add a new column to an existing SQLite table.

        :param table_name: Name of the table to modify.
        :param column_name: Name of the new column to add.
        :return: Integer count of not null lines
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # count Rill segments
        cursor.execute(f"SELECT * FROM {table_name} WHERE {column_name} IS NOT NULL")
        records = cursor.fetchall()
        conn.close()

        return len(records)
    
    
    def iterate_through_table(self, table_name, where_status = ""):
        """
        Iterate through the SQLite table.

        :param table_name: Name of the table to modify.
        :param column_name: Name of the new column to add.
        :return: Integer count of not null lines
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if where_status == "":
            sql_query = f"SELECT * FROM {table_name}"
        else:
            sql_query = f"SELECT * FROM {table_name} WHERE {where_status}"

        cursor.execute(sql_query)
        records = cursor.fetchall()
        
        return records
       
    
    def check_unique_record(self, table_name, new_dict):
        """
        Check if the record is unique before inserting it into table

        :param table_name: Name of the table to modify.
        :param new_dict: Dictionary to check if exists in the database
        :return: Integer count of not null lines
        """
        # create sql query
        where_parts = [f"{key} = ?" for key in new_dict.keys()]
        where_clause = " AND ".join(where_parts)
    
        sql_query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
        params = tuple(new_dict.values())                
        # print(sql_query)
    
        # check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(sql_query, params)
        
        # get the number of selected records
        records = cursor.fetchone()
        # print(records)
            
        if records[0] != 0:
            return True
        
        return False
    
    
    def check_unique_name(self, table_name, ID_column, record_name):
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql_query = f"SELECT 1 FROM {table_name} WHERE {ID_column} = ?"
        
        # Query to check if the scene name exists
        cursor.execute(sql_query, (record_name,))
        records = cursor.fetchone()
        
        if records[0] != 0:
            return True
        
        return False
    
    
    def load_json_as_dictionary(self, json_path):
        """
        Add a new column to an existing SQLite table.

        :param json_path: Name of the JSON to load.
        :return: Integer count of not null lines
        """
        
        # Opening JSON file
        json_file = open(json_path)
        # returns JSON object as a dictionary
        json_dict = json.load(json_file)
        # Closing file
        json_file.close()

        return json_dict
    
    
    def save_dictionary_as_JSON(self, dictionary_name, dictionary, json_path):
        """
        Add a new column to an existing SQLite table.

        :param json_path: Name of the table to modify.
        :return: Integer count of not null lines
        """
        
        save_dictionary = {}
        save_dictionary[f'{dictionary_name}'] = dictionary
        
        with open(json_path, "w") as write_file:
            json.dump(save_dictionary, write_file, indent=4)
            
            
    def load_table_name_and_columns(self, dictionary_name, json_path):
        """
        Add a new column to an existing SQLite table.

        :param dictionary_name: Name of the table to modify.
        :param json_path: Name of the table to modify.
        :return: table_content
        """
        
        dictionary = self.load_json_as_dictionary(json_path)
        # print(dictionary)
        try:
            table_content = dictionary[f'{dictionary_name}']
            print(f"Table name: {dictionary_name} and column names loaded")
            return table_content
        except Exception as e:
            print(f"Error loading {dictionary_name} from {json_path}: {e}")
            
    
    def add_dict_to_JSON(self, dictionary_name, new_dict, json_path):
        """
        Adds or updates a dictionary in a JSON file under a specified key.
    
        Parameters:
            dictionary_name (str): The name under which to store the new dictionary.
            new_dict (dict): The dictionary to store.
            json_path (str): Path to the JSON file.
        """
        
        # Load existing data if the file exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
    
        # Update or add new dictionary
        data[dictionary_name] = new_dict
    
        # Save back to JSON
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    
#%%

# import DB_management

table_name = 'Lists'
table_columns =  {
    "List_ID": "INTEGER PRIMARY KEY",
    "List_Name": "TEXT",
    "Folder": "TEXT",
    "Spatial_Reference": "TEXT",
    "X_Min": "FLOAT",
    "Y_Min": "FLOAT",
    "X_Max": "FLOAT",
    "Y_Max": "FLOAT",
    "Train_path": "TEXT",
    "Train_status": "TEXT",
    "Train_line_count": "REAL",
    "Pixel_number": "INTEGER",
    "Cell_size": "FLOAT",
    "Shift": "INTEGER",
    "Modification": "TEXT",
    "Just_others": "TEXT",
    "Width": "INTEGER",
    "Height": "INTEGER",
    "Dimension": "TEXT",
    "Segmentation": "TEXT",
    "Classification": "TEXT"
    }


# table_manag = DB_management(db_path)
# table_manag.save_dictionary_as_JSON(table_name, table_columns, json_columns_path)
# table_manag.drop_tables(['machine'])
# table_manag.create_table(table_name, table_columns)

#%%

class Folders_management:

    def __init__(self, db_path):
        """Initialize with the path to the database."""
        self.db_path = db_path

        # import other classes
        self.table_manag = DB_management(self.db_path)


    def get_rasters(self, raster_folder, file_extension, year):
        """
        Iterate through all JPG files in the folder and add them to DB.

        :raster_folder: Path to the folder with rasters.
        :file_extension: extension of rasters.
        :year: Year that should be in name of raster.
        """
        file_extension = ".tif"
        for filename in os.listdir(raster_folder):
            if filename.endswith(file_extension) and year in filename:

                new_dict = {}

                new_dict["List_Name"] = filename
                new_dict["Folder"] = raster_folder
                self.table_manag.create_record_from_dict("Lists", new_dict)
                
    def delete_inverted(self, model_folder):
        """
        Iterate through all JPG files in the folder and add them to DB.

        :raster_folder: Path to the folder with rasters.
        """
        class_list = ['Rill', 'NoRill']
                
        for item in class_list:
            
            # time management
            start = time.time()
            tot_time, C = 0, 1
            
            try:
                item_folder_name = os.path.join(model_folder, 'Mosaics', item)
                folder_content = os.listdir(item_folder_name)
            
            except Exception as e:
                logging.error("    Error loading content of folder {item_folder_name}", str(e))
            
            total = len(folder_content)
            i = 0
            
            for image in folder_content:
                if 'inverted' in image:
                    try:
                                                
                        file_path = os.path.join(item_folder_name, image)
                        os.remove(file_path)
                        print(f"{i}/{total} File {file_path} deleted")
                                                
                        # if i == 50:
                        #     break
                        
                    except Exception as e:
                        logging.error("    Error deleting image {file_path}", str(e))
            
                # time management
                end = time.time()
                tot_time += (end - start)
                        
                # estimated time left
                if i%100 == 0:
                    est_time = (len(folder_content) - i) * (tot_time/ C)
                    print()
                    if est_time > 86400:
                        print('Estimated time left:', round(est_time/86400, 2), 'days')
                    elif est_time > 3600:
                        print('Estimated time left:', round(est_time/3600, 2), 'hours')
                    elif est_time > 60:
                        print('Estimated time left:', round(est_time/60, 2), 'mins')
                    else:
                        print('Estimated time left:', round(est_time, 2), 'sec')
                    print()
                    tot_time, C = 0, 1
                    
                # counting mechanism
                i += 1
                C += 1
        
        print(f"Deleting of inverted images for {model_folder} done")
        
        
    def clean_model(self, model_folder):
        """
        Iterate through all JPG files in the folder and add them to DB.
        """
        
        # delete databases
        content_folder = os.path.join(model_folder, 'Mosaics')
        folder_content = os.listdir(content_folder)
        
        total = len(folder_content)
        i = 0
        
        for db in folder_content:
            if ".db" in db:
                    
                # time management
                start = time.time()
                tot_time, C = 0, 1
                
                try:
                    file_path = os.path.join(content_folder, db)
                    os.remove(file_path)
                    print(f"{i}/{total} File {file_path} deleted")
                
                except Exception as e:
                    logging.error("    Error deleting image {file_path}", str(e))
        
                # time management
                end = time.time()
                tot_time += (end - start)
                        
                # estimated time left
                if i%50 == 0:
                    est_time = (len(folder_content) - i) * (tot_time/ C)
                    print()
                    if est_time > 86400:
                        print('Estimated time left:', round(est_time/86400, 2), 'days')
                    elif est_time > 3600:
                        print('Estimated time left:', round(est_time/3600, 2), 'hours')
                    elif est_time > 60:
                        print('Estimated time left:', round(est_time/60, 2), 'mins')
                    else:
                        print('Estimated time left:', round(est_time, 2), 'sec')
                    print()
                    tot_time, C = 0, 1
                    
                # counting mechanism
                i += 1
                C += 1
                
        # delete images
        folders = ['Rill', 'NoRill', 'Rill_val', 'NoRill_val']
        
        for folder in folders:
            content_folder = os.path.join(model_folder, 'Mosaics', folder)
            folder_content = os.listdir(content_folder)
        
            total = len(folder_content)
            i = 0
            
            for image in folder_content:
                
                # time management
                start = time.time()
                tot_time, C = 0, 1
                
                try:
                    file_path = os.path.join(content_folder, image)
                    os.remove(file_path)
                    print(f"{i}/{total} File {file_path} deleted")
                
                except Exception as e:
                    logging.error("    Error deleting image {file_path}", str(e))
        
                # time management
                end = time.time()
                tot_time += (end - start)
                        
                # estimated time left
                if i%200 == 0:
                    est_time = (len(folder_content) - i) * (tot_time/ C)
                    print()
                    if est_time > 86400:
                        print('Estimated time left:', round(est_time/86400, 2), 'days')
                    elif est_time > 3600:
                        print('Estimated time left:', round(est_time/3600, 2), 'hours')
                    elif est_time > 60:
                        print('Estimated time left:', round(est_time/60, 2), 'mins')
                    else:
                        print('Estimated time left:', round(est_time, 2), 'sec')
                    print()
                    tot_time, C = 0, 1
                    
                # counting mechanism
                i += 1
                C += 1
        
        print(f"Deleting of inverted images for {model_folder} done")
        

#%%

# fold_manag = folders_management(db_path)
# fold_manag.get_rasters(r"S:\Projects\TACR-SIGMA-EROZE-ANN\cuzk_ortofoto\data_epsg5514", ".tif", "2011")



#%%


class raster_management:

    def __init__(self, db_path):
        """Initialize with the path to the database."""
        self.db_path = db_path

        # import other classes
        self.table_manag = DB_management(self.db_path)

    def raster_to_array(input_raster):
        """
        convert raster to numpy array, spit out numpy array
        using rasterio

        :input_raster: Path to raster.
        """
        ## export to numpy array
        # Open the TIFF file using rasterio
        with rasterio.open(input_raster) as src:
            # Read all bands as a NumPy array (shape: [bands, height, width])
            composite_array = src.read()




    def composite_arrays(array_list):
        """
        Composite arrays into one composite array

        :array_list: List of the individual arrays.
        """
        for item in array_list:
            if i == 0:
                composite_array = np.append(array_list[0], [array_list[i+1]], axis=0)
                i += 1

            else:
                composite_array = np.append(composite_array, [array_list[i+1]], axis=0)
                i += 1

        return composite_array


#%%

## calculate indexes



## merge input rasters to create raster for training segmentation

# RGB_raster =


# greyscale_array = np.around((0.3 * composite_array[red]) + (0.59 * composite_array[green]) + (0.11 * composite_array[blue]), 0)
# greyscale_array = greyscale_array.astype('int')
# composite_array = np.append(composite_array, [greyscale_array], axis=0)
# compo_dimensions = composite_array.shape

##


## segment raster into training data and create db of segments













