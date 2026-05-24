import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper


class ImportBayoMOT(bpy.types.Operator, ExportHelper):
    '''Import MOT Data.'''
    bl_idname = "import_scene.bayo_mot_data"
    bl_label = "Import MOT Data"
    bl_options = {'PRESET'}
    filename_ext = ".mot"
    filter_glob: StringProperty(default="*.mot", options={'HIDDEN'})



    def execute(self, context):
        from . import mot_importer
        mot_importer.ImportMOT(self.filepath)
        return {'FINISHED'}
    