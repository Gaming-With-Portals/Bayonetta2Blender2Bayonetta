import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper


class ImportMadWorldMDB(bpy.types.Operator, ExportHelper):
    '''Import MDB Data.'''
    bl_idname = "import_scene.bayo_mdb_data"
    bl_label = "Import MDB Data"
    bl_options = {'PRESET'}
    filename_ext = ".mdb"
    filter_glob: StringProperty(default="*.mdb", options={'HIDDEN'})

    #reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)


    def execute(self, context):
        f = open(self.filepath, "rb")
        tag = f.read(2)
        if (tag == b"JK"):
            from .mdb_importer import ImportMDB
            ImportMDB(self.filepath)

        else:
            print(f"[!] Unsupport MDB version (what): {tag.decode()}")
            return {"CANCELLED"}

        return {"FINISHED"}