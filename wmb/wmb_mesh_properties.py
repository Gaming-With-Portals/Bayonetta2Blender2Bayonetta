import bpy

class BayoMeshDataProperty(bpy.types.PropertyGroup):
    bayonetta_2 : bpy.props.BoolProperty(
        name="Bayonetta 2?",
        description="Is it a Bayonetta 2 mesh",
        default=0
    )
    
    flags: bpy.props.IntProperty(
        name="Flags",
        description="Material Flags",
        default=0
    )
