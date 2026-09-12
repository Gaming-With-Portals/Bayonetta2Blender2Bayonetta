import bpy

class BayoBatchDataProperty(bpy.types.PropertyGroup):
    bayonetta_2 : bpy.props.BoolProperty(
        name="Bayonetta 2?",
        description="Is it a Bayonetta 2 mesh",
        default=0
    )
    
    flags: bpy.props.IntProperty(
        name="Flags",
        description="Flags",
        default=0
    )
