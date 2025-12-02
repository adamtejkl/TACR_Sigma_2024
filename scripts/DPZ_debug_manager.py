# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:40:15 2025

@author: Adam Tejkl
"""

# import additional libraries
import inspect
import sys
import os

# import my modules
# sys.path.append(r'K:\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\02_Tejkl')
sys.path.append(r'\\data.fsv.cvut.cz\shares\K143\Private\_PROJEKTY\2024_TACR_DPZ_junior\01_reseni_projektu\04_plug_in\scripts')
from DPZ_model import DB_management

#%%

class Debug_Manager():
    
    
    def __init__(self, root, db_path, config_json_path):
        """
        debug_level:
            0 = Silent
            1 = Minimal
            2 = Medium
            3 = Loud
        """
        
        self.root = root
        self.db_path = db_path
        self.config_json_path = config_json_path
        
        self.table_manag = DB_management(self.db_path)
    
        # load debug level from config file
        self.debug_dict = self.table_manag.load_table_name_and_columns('debug', self.config_json_path)
        self.debug_level = self.debug_dict['debug_level']
        self.debug_flags = self.debug_dict['debug_flags']
        
        
#%%

#
##
#

    def _get_location(self):
        """Internal helper function to get the line number and filename"""
        frame = inspect.currentframe()
        # step back two frames: _get_location → debug_print → caller
        caller = frame.f_back.f_back
        filename = caller.f_code.co_filename
        line = caller.f_lineno
        return filename, line


#%%
    
    #
    ## debug printing line
    #
    
    def debug(self, message, statement_level=1, end="\n"):
        """Prints debug information based on debug level.
        Inputs
        :param message (text): Message to be printed.
        :param debug_level (int):
        :param end (int): ending string, default newline ("\n"). Set to "" for no newline.
    
        Returns
        None.
        """
        
        if self.debug_level >= statement_level:
            # Get caller frame
            frame = inspect.currentframe().f_back
            func_name = frame.f_code.co_name
            line_no = frame.f_lineno
            
            filename, line = self._get_location()
            location = f"[{filename}:{line}]"
    
            print(f"   {location} {func_name} (line {line_no}): {message}", end = end)
            
            
#%%

#
##
#

    def debug_action(self, flag_name, action, *args, **kwargs):
        """
        Runs the given action only if a debug flag is enabled.
    
        flag_name: string naming the flag
        action: a function to execute
        args/kwargs: forwarded to the action function
        """
        
        # returns the value of the key, if it does not exist throws False
        if self.debug_flags.get(flag_name, False):
            
            filename, line = self._get_location()
            location = f"[{filename}:{line}]"
            
            return action(*args, **kwargs) 


#%%

model_number = 'zajmove_obdelniky'
folder = r"\\backup.fsv.cvut.cz\Users\tejklada\DPZ_junior_Tejkl\Models"

root = os.path.join(folder, f'model_{model_number}')
db_path = os.path.join(root, f'model_{model_number}.db')
config_json_path = os.path.join(root, 'config.json')

debug_manag = Debug_Manager(root, db_path, config_json_path)
        
        
print(debug_manag.debug_level)
print(debug_manag.debug_flags)
        
        
debug_manag.debug_action('print_all', print, 'tohle')
        
        
        
        