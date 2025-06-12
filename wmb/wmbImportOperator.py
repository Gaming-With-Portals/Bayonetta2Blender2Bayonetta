import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper


class ImportBayoWMB(bpy.types.Operator, ExportHelper):
    '''Import WMB Data.'''
    bl_idname = "import_scene.bayo_wmb_data"
    bl_label = "Import WMB Data"
    bl_options = {'PRESET'}
    filename_ext = ".wmb"
    filter_glob: StringProperty(default="*.wmb", options={'HIDDEN'})

    reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)

    def execute(self, context):
        from . import wmb_importer
        return  wmb_importer.ImportWMB(self.filepath)
    
class ExportBayoWMB(bpy.types.Operator, ExportHelper):
    '''Export WMB Data.'''
    bl_idname = "export.bayo_wmb_data"
    bl_label = "Export WMB File"
    bl_options = {'PRESET'}
    filename_ext = ".wmb"
    filter_glob: StringProperty(default="*.wmb", options={'HIDDEN'})


    def execute(self, context):
        from . import wmb_exporter
        return  wmb_exporter.export(self.filepath)

