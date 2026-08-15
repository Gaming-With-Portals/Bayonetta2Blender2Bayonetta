from ...structwrapper import BinReader
import bpy
import os
import math

def align(offset, alignment):
    return offset if offset % alignment == 0 else offset + (alignment - (offset % alignment))

def ImportMDB(path):
    rf = open(path, "rb")
    f = BinReader(rf, True) # call me the wii the way I... uhhh idk
    f.update_endianess_flag()
    # its 10pm rn i gotta wake up in like 6 hours

    target_col = "MDB" # I don't care
    if (target_col=="MDB"):
        if ("MDB" not in bpy.data.collections):
            wmb_collection = bpy.data.collections.new("MDB")
            bpy.context.scene.collection.children.link(wmb_collection)
        else:
            wmb_collection = bpy.data.collections["MDB"]
    else:
        wmb_collection = bpy.data.collections[target_col]

    wmb_name = os.path.splitext(os.path.basename(path))[0] # yo wtf is a wmb lol - the madworld devs, probably

    model_collection = bpy.data.collections.new(wmb_name)

    wmb_collection.children.link(model_collection)

    f.read_u16()
    vertex_flags = f.read_u16()
    num_vertex_pos = f.read_u32()
    vertex_pos_offset = f.read_u32()
    num_normals = f.read_u32()
    normals_offset = f.read_u32()
    num_uvs = f.read_u32()
    uv_offset = f.read_u32()

    num_colors = f.read_u32()
    colors_offset = f.read_u32()

    num_bone_palette = f.read_u32()
    bone_palette_offset = f.read_u32()
    num_bones = f.read_u32()
    parents_offset = f.read_u32()
    bone_rel_pos_offset = f.read_u32()
    bone_pos_offset = f.read_u32()

    num_materials = f.read_u32()
    material_offset = f.read_u32()

    f.read_u32()

    meshes_offset = f.read_u32()
    position_divisor_bit = f.read_u16()
    uv_divisor_bit = f.read_u16()
    f.read_u32()

    f.read_u32()
    f.read_u32()

    bone_name_offset = f.read_u32()
    bone_palette_reference_count_offset = f.read_u32()
    print(f.tell())
    num_meshes = f.read_u16()


    print(meshes_offset)


    f.seek(meshes_offset)
    for i in range(num_meshes):

        

        current_pos = f.tell() # yeah ok bro
        next_mesh_offset = f.read_u32()
        f.read_u8()
        flags = f.read_u8()
        print(next_mesh_offset)

        f.seek(current_pos+20)
        index_buffer_size = f.read_u32()
        index_buffer_count = f.read_u32()


        f.seek(current_pos+32)
        if (vertex_flags & 0x200):
            name = f.read(0x3c).decode().replace("\x00", "")
            header_size = 96
        else:
            name = f.read(0x20).decode().replace("\x00", "")
            header_size = 64

        index_buffers = []
        f.seek(align(current_pos+header_size, 0x20)) # ts cannot be real
        for x in range(index_buffer_count):
            index_buffers.append([])
            ib_type = f.read_u8()
            ib_count = f.read_u16()
            print(f"[>] Reading {ib_count} indices")

            for _ in range(ib_count):
                index_data = [0, 0, 0, 0]

                index_data[0] = f.read_u16()
                if (num_normals > 0 and not (flags & 0x4)):
                    index_data[1] = f.read_u16()
                if (num_colors > 0 and not (flags & 0x1)):
                    index_data[2] = f.read_u16()
                if (num_uvs > 0 and not (flags & 0x8)):
                    index_data[3] = f.read_u16()

                index_buffers[x].append(index_data)

            

        
        for x in range(index_buffer_count):
            object_name = f"{i}-{name}-{x}"

            mesh = bpy.data.meshes.new(object_name)

            obj = bpy.data.objects.new(object_name, mesh)


            model_collection.objects.link(obj)
            obj.rotation_euler = (math.radians(90), 0, 0)


        f.seek(current_pos+next_mesh_offset)