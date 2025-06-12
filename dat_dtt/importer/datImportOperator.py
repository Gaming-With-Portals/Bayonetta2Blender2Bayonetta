import os

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

# Add this import statement at the top of the file
from ...utils.visibilitySwitcher import enableVisibilitySelector
from ...utils.util import setExportFieldsFromImportFile, ShowMessageBox
from ...consts import DAT_EXTENSIONS

def ImportData(only_extract, filepath, transform=None):
    print("Importing data...")
    extension = os.path.splitext(filepath)[1]
    
    # This is a much better naming scheme
    head = os.path.split(filepath)[0]
    filename_with_extension = os.path.split(filepath)[1]
    filename_without_extension = filename_with_extension[:-4]
    extract_dir = os.path.join(head, 'bayo2blender_extracted')
    
    dat_filepath = os.path.join(head, filename_without_extension + '.dat')
    dtt_filepath = os.path.join(head, filename_without_extension + '.dtt') 
    
    dtt_filename = "" # Initalization
    dat_filename = ""
    
    from . import dat_unpacker
    print("DAT Path: " + dat_filepath)
    if os.path.isfile(dat_filepath):
        dat_filename = dat_unpacker.main(dat_filepath, os.path.join(extract_dir, filename_without_extension + '.dat'), dat_filepath)
    print("DTT Path: " + dtt_filepath)
    if os.path.isfile(dtt_filepath):
        dtt_filename = dat_unpacker.main(dtt_filepath, os.path.join(extract_dir, filename_without_extension + '.dtt'), dtt_filepath)
    
    if (dat_filename == "" and dtt_filename == ""):
        print("I have no idea how you managed to select a DAT or DTT if you had neither")
    if (not os.path.isfile(dat_filepath) and os.path.isfile(dtt_filepath)):
        print("Did you just type random nonsense into the file select?")
    if (dat_filename == False and dtt_filename == False):
        print("Apparently, both the DAT and DTT associated with your selection are empty.")
        
    
    scr_mode = False
    wmb_mode = False
    wmb_ext = ".dat"
    scr_ext = ".dat"
    
    if (dtt_filename == False):
        # I dunno what this does but in the forefathers I trust
        dtt_filename = dat_filename[:10]
        
    wmb_files = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dat')) if os.path.splitext(x)[1] == ".wmb"]
    if (len(wmb_files) == 0):
        # Last chance to show yourself
        print("Attempting to find WMB in DTT (fruitless)")
        if (os.path.exists(os.path.join(extract_dir, filename_without_extension + '.dtt'))):
            wmb_files = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dtt')) if os.path.splitext(x)[1] == ".wmb"]
            wmb_ext = ".dtt"

    scr_files = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dat')) if os.path.splitext(x)[1] == ".scr"]

    if (len(wmb_files) > 0):
        wmb_mode = True
    elif (len(scr_files) == 0):
        # Last chance to show yourself
        print("Attempting to find SCR in DTT (fruitless)")
        if os.path.exists(os.path.join(extract_dir, filename_without_extension + '.dtt')):
            scr_files = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dtt')) if os.path.splitext(x)[1] == ".scr"]
            scr_ext = ".dtt"

    if (len(scr_files) > 0):
        scr_mode = True

    print("SCR Files: " + str(scr_files))
    print("WMB Files: " + str(wmb_files))

    # WTA/WTP
    wtbPath = os.path.join(extract_dir, filename_without_extension + '.dat', filename_without_extension + '.wtb')
  
    if os.path.isfile(wtbPath):
        texturesExtractDir = os.path.join(extract_dir, filename_without_extension + '.dat', "textures")
        from ...wta_wtp.importer import wtpImportOperator
        wtpImportOperator.extractFromWta(wtbPath, wtbPath, texturesExtractDir)

    if only_extract:
        return {'FINISHED'}

    if wmb_mode:
        # WMB
        if filename_without_extension + ".wmb" in wmb_files:
            wmb_files = [filename_without_extension + ".wmb"]
        wmb_filepath = os.path.join(extract_dir, filename_without_extension + wmb_ext, wmb_files[0])
        print("WMB Path: " + wmb_filepath)
        from ...wmb import wmb_importer
        wmb_importer.ImportWMB(wmb_filepath, os.path.join(extract_dir, filename_without_extension + '.dat', "textures"))

    setExportFieldsFromImportFile(filepath, True)
    enableVisibilitySelector()
    
    
    
    
    
    return {'FINISHED'}
    


class ImportNierDtt(bpy.types.Operator, ImportHelper):
    '''Load a DTT (and DAT) File.'''
    bl_idname = "import_scene.bayo_dtt_data"
    bl_label = "Import DTT (and DAT) Data"
    bl_options = {'PRESET'}
    filename_ext = ".dtt"
    filter_glob: StringProperty(default="*.dtt", options={'HIDDEN'})

    reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)
    bulk_import: bpy.props.BoolProperty(name="Bulk Import All DTT/DATs In Folder (Experimental)", default=False)
    only_extract: bpy.props.BoolProperty(name="Only Extract DTT/DAT Contents. (Experimental)", default=False)

    def execute(self, context):
        print("Unpacking", self.filepath)
        if self.bulk_import:
            folder = os.path.split(self.filepath)[0]
            for filename in os.listdir(folder):
                if filename[-4:] == '.dtt':
                    try:
                        filepath = os.path.join(folder, filename)
                        print("\nImporting", filepath)
                        ImportData(self.only_extract, filepath)
                    except:
                        print('ERROR: FAILED TO IMPORT', filename)
            return {'FINISHED'}

        else:
            return ImportData(self.only_extract, self.filepath)
        


class ImportNierDat(bpy.types.Operator, ImportHelper):
    '''Load a DAT File.'''
    bl_idname = "import_scene.bayo_dat_data"
    bl_label = "Import DAT/DTT Data"
    bl_options = {'PRESET'}
    filename_ext = ".dat;.dtt"
    filter_glob: StringProperty(default=";".join([f"*{ext}" for ext in DAT_EXTENSIONS]), options={'HIDDEN'})

    reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)
    bulk_import: bpy.props.BoolProperty(name="Bulk Import All DTT/DATs In Folder", default=False)
    only_extract: bpy.props.BoolProperty(name="Only Extract DTT/DAT Contents", default=False)

    def doImport(self, onlyExtract, filepath):
        head = os.path.split(filepath)[0]
        tail = os.path.split(filepath)[1]
        ext = tail[-4:]
        tailless_tail = tail[:-4]
        dat_filepath = os.path.join(head, tailless_tail + ext)
        extract_dir = os.path.join(head, 'bayo2blender_extracted')
        from . import dat_unpacker
        if os.path.isfile(dat_filepath):
            dat_unpacker.main(dat_filepath, os.path.join(extract_dir, tailless_tail + ext), dat_filepath)   # dat

        if onlyExtract:
            return {'FINISHED'}

        setExportFieldsFromImportFile(filepath, True)

        return {'FINISHED'}

    def execute(self, context):
        print("Unpacking", self.filepath)
        firstModel = ImportData(self.only_extract, self.filepath)
        if self.bulk_import:
            folder = os.path.split(self.filepath)[0]
            for filename in os.listdir(folder):
                if filename[-4:] == '.dat':
                    try:
                        filepath = os.path.join(folder, filename)
                        if filepath != self.filepath: # Already got that one
                            ImportData(self.only_extract, filepath)
                    except:
                        print('ERROR: FAILED TO IMPORT', filename)
            return {'FINISHED'}
        return firstModel

