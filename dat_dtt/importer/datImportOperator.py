import os

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

# Add this import statement at the top of the file
from ...utils.visibilitySwitcher import enableVisibilitySelector
from ...utils.util import setExportFieldsFromImportFile, ShowMessageBox
from ...consts import DAT_EXTENSIONS
from ...wta_wtp.pg_texture import extractTextures

def ImportData(only_extract, filepath, transform=None, isStageSubmesh=False):
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
    lyt_mode = False
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

    lyt_files = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dat')) if os.path.splitext(x)[1] == ".lyt"]

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

    if filename_without_extension + ".wmb" in wmb_files: # For b1, attempt to find a wmb with the dat name
        wmb_files = [filename_without_extension + ".wmb"]
    if "pl0010.wmb" in wmb_files: # Exception for bayonetta because it's weird
        wmb_files = ["pl0010.wmb"] 

    # WTA/WTP
    wtbFiles = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dat')) if os.path.splitext(x)[1] == ".wtb"]
    #wtbPath = os.path.join(extract_dir, filename_without_extension + '.dat', filename_without_extension + '.wtb')
    wtaFiles = [x for x in os.listdir(os.path.join(extract_dir, filename_without_extension + '.dat')) if os.path.splitext(x)[1] == ".wta"]
    #wtaPath = os.path.join(extract_dir, filename_without_extension + '.dat', filename_without_extension + '.wta')
  
    if (len(wtbFiles) > 0):
        print("Textures: WTB")
        for wtbPath in wtbFiles:
            if len(wmb_files) > 0:
                if (not wtbPath.startswith(os.path.splitext(wmb_files[0])[0])):
                    continue
            elif len(scr_files) > 0:
                if (not wtbPath.startswith(os.path.splitext(scr_files[0])[0])):
                    continue

            if "tex" in os.path.basename(wtbPath):
                print("Skipping 'tex' entry!")
                continue
            
            wtbPath = os.path.join(extract_dir, filename_without_extension + '.dat', wtbPath)
            texturesExtractDir = os.path.join(extract_dir, filename_without_extension + '.dat', "textures")
            extractTextures(wtbPath, wtbPath, texturesExtractDir)

    if (len(wtaFiles) > 0):
        print("Textures: WTA")
        wtaPath = os.path.join(extract_dir, filename_without_extension + '.dat', wtaFiles[0])

        wtpPath = os.path.join(extract_dir, filename_without_extension + '.dat', os.path.splitext(wtaPath)[0]+".wtp")
        if (not os.path.isfile(wtpPath)):
            wtpPath = os.path.join(extract_dir, filename_without_extension + '.dtt', os.path.splitext(wtaPath)[0]+".wtp")

        if (os.path.isfile(wtpPath)):
            texturesExtractDir = os.path.join(extract_dir, filename_without_extension + '.dat', "textures")
            extractTextures(wtaPath, wtpPath, texturesExtractDir)
        else:
            print("Couldn't find WTP, skipping textures")

    if (len(lyt_files) > 0):
        lyt_mode = True


    if only_extract:
        return {'FINISHED'}

    if wmb_mode:
        # WMB
        wmb_filepath = os.path.join(extract_dir, filename_without_extension + wmb_ext, wmb_files[0])
        print("WMB Path: " + wmb_filepath)
        from ...wmb import wmb_importer
        if (isStageSubmesh):
            wmb_importer.ImportWMB(wmb_filepath, os.path.join(extract_dir, filename_without_extension + '.dat', "textures"), True, True, target_col="LYT")
        else:
            wmb_importer.ImportWMB(wmb_filepath, os.path.join(extract_dir, filename_without_extension + '.dat', "textures"), True, True)

    elif scr_mode:
        # WMB
        scr_filepath = os.path.join(extract_dir, filename_without_extension + scr_ext, scr_files[0])
        print("SCR Path: " + scr_filepath)
        from ...scr import scr_importer
        scr_importer.ImportSCR(scr_filepath)

    elif lyt_mode:
        # LYT
        lyt_filepath = os.path.join(extract_dir, filename_without_extension + ".dat", lyt_files[0])
        print("LYT Path: " + lyt_filepath)
        from ...scr import lyt_importer
        lyt_importer.ImportLYT(lyt_filepath)

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

