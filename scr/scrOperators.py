import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper


class ImportBayoSCR(bpy.types.Operator, ExportHelper):
    '''Import SCR Data.'''
    bl_idname = "import_scene.bayo_scr_data"
    bl_label = "Import SCR Data"
    bl_options = {'PRESET'}
    filename_ext = ".scr"
    filter_glob: StringProperty(default="*.scr", options={'HIDDEN'})

    #reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)

    def execute(self, context):
        from . import scr_importer
        return scr_importer.ImportSCR(self.filepath)
    

class ImportVanqLYT(bpy.types.Operator, ExportHelper):
    '''Import LYT Data.'''
    bl_idname = "import_scene.bayo_lyt_data"
    bl_label = "Import LYT Data"
    bl_options = {'PRESET'}
    filename_ext = ".lyt"
    filter_glob: StringProperty(default="*.lyt", options={'HIDDEN'})

    #reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)

    def execute(self, context):
        from . import lyt_importer
        return lyt_importer.ImportLYT(self.filepath)