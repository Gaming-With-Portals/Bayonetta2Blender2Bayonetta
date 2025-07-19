import bpy
import struct
from collections import defaultdict
import bmesh
from io import BufferedReader
from mathutils import Vector, Matrix
import numpy as np

local_bone_to_id_map = {}
global_name_to_local_id = {} # for those stupid ass vertex groups

def align(offset, alignment):
    return offset if offset % alignment == 0 else offset + (alignment - (offset % alignment))

def float_to_half_bytes(f):
    h = np.float16(f)
    return h.tobytes()

def encode_parts_index_no_table(parts_no_index_map):
    # Credits to Skyth
    l1_table = [0xFFFF] * 16
    l2_tables = []
    l3_tables = []

    l2_index_map = {}
    l3_index_map = {}
    l3_data = {}

    for parts_no, parts_index in parts_no_index_map:
        l1 = (parts_no >> 8) & 0xF
        l2 = (parts_no >> 4) & 0xF
        l3 = parts_no & 0xF

        if l1 not in l2_index_map:
            l2_index_map[l1] = len(l2_tables)
            l2_tables.append([0xFFFF] * 16)

        l2_table = l2_tables[l2_index_map[l1]]

        l3_key = (l1, l2)
        if l3_key not in l3_index_map:
            l3_index_map[l3_key] = len(l3_tables)
            l3_data[l3_key] = [0xFFF] * 16
            l3_tables.append(l3_data[l3_key])

        parts_indices = l3_data[l3_key]
        if parts_indices[l3] != 0xFFF:
            raise ValueError("{} was somehow already added to the table!".format(parts_no))

        parts_indices[l3] = parts_index

    l2_offset_base = 16
    l3_offset_base = l2_offset_base + len(l2_tables) * 16

    for l1, l2_idx in l2_index_map.items():
        l1_table[l1] = l2_offset_base + l2_idx * 16

    for (l1, l2), l3_idx in l3_index_map.items():
        l2_table = l2_tables[l2_index_map[l1]]
        l2_table[l2] = l3_offset_base + l3_idx * 16

    table = l1_table
    for l2 in l2_tables:
        table.extend(l2)
    for l3 in l3_tables:
        table.extend(l3)

    return table


class WMBVertexChunk:
    def __init__(self, children):
        self.vertex_infos = []
        self.exvertex_infos = []
        self.total_vertices = 0
        self.num_mapping = 2
        self.num_color = 1
        
        vertex_ticker = 0
        for obj in children:
            if obj.type != 'MESH':
                continue

            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()

            eval_mesh.calc_tangents(uvmap=eval_mesh.uv_layers.active.name)

            self.total_vertices += len(eval_mesh.vertices)
            sorted_loops = sorted(eval_mesh.loops, key=lambda loop: loop.vertex_index)


            uv_layer = eval_mesh.uv_layers.active
            loop_map = {} 

            for loop in eval_mesh.loops:
                vidx = loop.vertex_index
                if vidx not in loop_map:
                    loop_map[vidx] = uv_layer.data[loop.index].uv.copy() 

            obj["vertex_start"] = vertex_ticker
            for vertex in eval_mesh.vertices:
                vertex_ticker+=1
                pos = vertex.co
                normal = vertex.normal  # vertex normal (not loop normal)
                
                uv = loop_map.get(vertex.index, (0.0, 0.0))
                MAX_WEIGHTS = 4
                bone_weights = []
                bone_indices = []
                for g in vertex.groups:
                    group_index = g.group
                    weight = g.weight
                    # Group indices directly map to bone indices in the batch
                    bone_weights.append(weight)
                    bone_indices.append(group_index)

                bone_data = sorted(zip(bone_weights, bone_indices), reverse=True)[:MAX_WEIGHTS]
                weights, indices = zip(*bone_data) if bone_data else ([], [])
                weights = list(weights) + [0.0] * (MAX_WEIGHTS - len(weights))
                indices = list(indices) + [0] * (MAX_WEIGHTS - len(indices))

                normal = obj.matrix_world.to_3x3() @ vertex.normal
                normal.normalize()


                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]

                vertex_info = []
                vertex_info.append(pos.copy())
                vertex_info.append((normal.x, normal.z, -normal.y))
                vertex_info.append(0)  # Tangents
                vertex_info.append(tuple(indices))
                vertex_info.append(tuple(int(w * 255) for w in weights))
                vertex_info.append(uv.copy())
                self.vertex_infos.append(vertex_info)
            obj["vertex_end"] = vertex_ticker

def getBoneID(boneName):
    return int(boneName[4:]) # This will be important later

def getLocalBoneID(bone):
    if not bone in local_bone_to_id_map:
        print("What the fuck, this will crash and not work in game and thank goodness because how did you do this ")

    return int(local_bone_to_id_map[bone])


class WMBBoneParents:
    def __init__(self, arm_obj):
        self.bone_map = {}
        for bone in arm_obj.data.bones:
            if bone.parent:
                self.bone_map[getLocalBoneID(bone)] = getLocalBoneID(bone.parent)
            else:
                self.bone_map[getLocalBoneID(bone)] = -1

class WMBBonePosition:
    def __init__(self, arm_obj):
        print("[>] Generating bone position data...")
        self.bone_rel_map = {}
        self.bone_pos_map = {}

        for bone in arm_obj.data.bones:
            parent_pos = Vector((0, 0, 0))
            if bone.parent:
                parent_pos = bone.parent.head_local
            
            bone_rel_pos = bone.head_local - parent_pos
            bone_abs_pos = bone.head_local

            self.bone_rel_map[getLocalBoneID(bone)] = (bone_rel_pos.x, bone_rel_pos.y, bone_rel_pos.z)
            self.bone_pos_map[getLocalBoneID(bone)] = (bone_abs_pos.x, bone_abs_pos.y, bone_abs_pos.z)



class WMBBoneIndexTranslateTable:
    def __init__(self, arm_obj):
        translate_table_food = []
        for bone in arm_obj.data.bones:
            translate_table_food.append((bone["id"], getLocalBoneID(bone)))

        self.data = encode_parts_index_no_table(sorted(translate_table_food, key = lambda x: x[1]))
        self.size = len(self.data) * 2

class WMBInverseKinetic:
    def __init__(self, arm_obj):
        self.enabled = arm_obj["inverse_kinematics"]
        self.size = 0

        if (self.enabled):
            self.size+=8 # Header
            self.count = arm_obj["ik_count"]
            self.data = arm_obj["ik_unk"]
            self.offset = arm_obj["ik_offset"]
            self.structures = []
            for i in range(self.count):
                self.size+=16
                self.structures.append(arm_obj["ik_structure_" + str(i)])

class WMBBoneSymmetries:
    def __init__(self, arm_obj):
        # TODO: Not perfect
        self.enabled = False
        self.sym_map = {}
        if (arm_obj["bone_symmetries"]):
            self.enabled = True
            
            for bone in arm_obj.data.bones: # this is such a terrible way of doing this lmao
                pos = bone.head_local
                for bone_2 in arm_obj.data.bones:
                    if bone.name == bone_2.name:
                        continue

                    pos_2 = bone_2.head_local
                    if (pos[0] == 0 or pos_2[0] == 0): # Skip center bones
                        continue


                    if abs(-pos[0] - pos_2[0]) < 0.0001:
                        self.sym_map[getLocalBoneID(bone)] = getLocalBoneID(bone)

class WMBBoneFlags:
    def __init__(self, arm_obj):
        self.enabled = False
        self.flag_map = {}
        if (arm_obj["bone_flags"]):
            bpy.context.view_layer.objects.active = arm_obj
            self.enabled = True
            for bone in arm_obj.data.bones:
                if "flags" in bone:
                    self.flag_map[getLocalBoneID(bone)] = bone["flags"]
                else:
                    self.flag_map[getLocalBoneID(bone)] = 5

class WMBMaterial(): # Good enough for a direct port 
    def __init__(self):
        self.id = 0
        self.flag = 0
        self.size = 0
        self.type = 0
        self.data = []

    def fetch_size(self):
        return self.size
    
    def write(self, f):
        f.write(struct.pack("<H", self.type))
        f.write(struct.pack("<H", self.flag))
        for i in self.data: # Write raw data
            f.write(struct.pack("<i", i))


class WMBMaterialBlob:
    def __init__(self, arm_obj):
        unique_mats = set()
        self.materials = []

        for obj in arm_obj.children:
            if obj.type != 'MESH':
                continue
            for slot in obj.material_slots:
                if slot.material:
                    unique_mats.add(slot.material)

        sorted_mats = sorted(unique_mats, key=lambda m: int(m.name.rsplit("_", 1)[-1]))

        for mat in sorted_mats:
            emat = WMBMaterial()
            emat.id = int(mat.name.rsplit("_", 1)[-1])
            emat.size = int(mat["size"])
            emat.flag = int(mat["flags"])
            emat.type = int(mat["type"])
            emat.data = mat["data"]
            self.materials.append(emat)

        self.material_count = len(self.materials)

        self.offsets = []
        offset_ticker = 0
        for mat in self.materials:
            self.offsets.append(offset_ticker)
            offset_ticker+=mat.fetch_size()

        self.total_size = offset_ticker


class WMBMeshBlob():
    def __init__(self, arm_obj):
        self.mesh_count = 0
        self.offsets = []
        self.meshes = []

        
        for obj in arm_obj.children:
            if obj.type != 'MESH':
                continue
            name_parts = obj.name.split("-")
            if (len(name_parts) == 3):
                if (int(name_parts[2]) == 0):
                    self.mesh_count+=1

class WMBBatch():
    def __init__(self, parent, obj):
        self.batch_idx = 0
        self.id = parent.mesh_id
        self.flags = 32769
        self.exmaterial_id = 0
        self.material_id = int(obj.material_slots[0].name.rsplit("_", 1)[-1])
        self.has_bone_refs = 0
        self.vertex_start = obj["vertex_start"]
        self.vertex_end = obj["vertex_end"]
        self.primitive_type = 4
        self.indice_offset = 128


        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)  # Convert to tris
        bm.to_mesh(mesh)
        bm.free()
        mesh.calc_loop_triangles()
        self.indices = []
        for tri in mesh.loop_triangles:
            self.indices.extend([
                # Winding order needs to be flipped to appear correct in game.
                tri.vertices[2] + self.vertex_start,
                tri.vertices[1] + self.vertex_start,
                tri.vertices[0] + self.vertex_start,
            ])

        self.required_bones = []
        for group in obj.vertex_groups:
            self.has_bone_refs = 1
            self.required_bones.append(global_name_to_local_id[group.name])

    def fetch_size(self):
        size = 256
        size+=len(self.indices)*2

        return size
        


class WMBMesh():
    def __init__(self, arm_obj, obj, ofticker):
        name_parts = obj.name.split("-")
        self.name = name_parts[1]
        self.exdata = obj["data"]
        self.mesh_id = int(name_parts[0])
        bpy_batches = []
        self.batches = []
        

        for obj in arm_obj.children:
            if obj.type != 'MESH':
                continue
            name_parts = obj.name.split("-")
            if (int(name_parts[0]) == self.mesh_id):
                bpy_batches.append((int(name_parts[2]), obj))

        bpy_batches.sort(key=lambda x: x[0])
        bpy_batches = [obj for _, obj in bpy_batches]

        self.batch_count = len(bpy_batches)
        self.flags = -2147483648
        self.batch_offset_offset = 128
        self.bounding_box_infos = 1

        self.batch_start_offset = align(4 * self.batch_count, 32)
        offset = self.batch_start_offset
        self.batch_offsets = []
        for batch_obj in bpy_batches:
            tmp_batch = WMBBatch(self, batch_obj)
            self.batches.append(tmp_batch)
            self.batch_offsets.append(offset)
            offset+=tmp_batch.fetch_size()

    def fetch_size(self):
        size = self.batch_start_offset
        for batch in self.batches:
            size += batch.fetch_size()
        return size


class WMBDataGenerator:
    def __init__(self, colName="WMB"):
        
        wmb_collection =  bpy.context.view_layer.layer_collection.children[colName]
        sub_collection = [x for x in wmb_collection.children if x.is_visible][0]
        arm_obj = sub_collection.collection.objects[0]
        for child in arm_obj.children:
            if child.type != 'MESH':
                continue
            
        # I do need custom local bone indexes. And global ones, wanna know why? "I have a dream." That one day, every person who uses this plugin will control their OWN armatures
        # A land of the TRULY free, dammit, a plugin of ACTION, not documentation, ruled by CREATIVE, not edge cases. Where the code changes to suit the modder. 
        # Not the other way around. Where power and hcoice are back where they belong, in the hands of the modders! Where every modder is free to mod, and customize, for himself!
        # Fuck all these limp-dick tools and chicken-shit 'it works on my machine'. Fuck this 24/7 internet spew of trivia and CruelerThanDAT bullshit. Fuck patchers, fuck the media
        # fuck all of it! Bayonetta models are diseased, rotten to the core, there's no saving it -- we need to pull it out by the roots, wipe the slate clean. BURN IT DOWN!
        # And from the ashes, a new model modding tool will be born, evolved, but untamed! The weak will be purged, and the strongest will thrive -- free to mod as they see fit,
        # they'll make modding great again!
        for i, bone in enumerate(sorted(arm_obj.data.bones, key=lambda x: x["id"])):
            local_bone_to_id_map[bone] = i
            global_name_to_local_id[bone.name] = i
            bone["read_only_local_index"] = i

        offset_ticker = 0
        self.header_offset = 0
        self.header_size = 128
        self.vtx_format = sub_collection.collection["vertex_format"]
        self.bone_count = len(arm_obj.data.bones)

        self.has_that_one_fucked_vertex_format = False
        if self.vtx_format == 0x4000001F:
            self.has_that_one_fucked_vertex_format = True

        offset_ticker += self.header_size
        offset_ticker = align(offset_ticker, 32)

        ## -- VERTEX CHUNK A --
        self.offset_vertexes = offset_ticker
        self.vertex_data = WMBVertexChunk(arm_obj.children)
        offset_ticker += self.vertex_data.total_vertices * 32
        offset_ticker = align(offset_ticker, 32)

        self.offset_ex_vertexes = offset_ticker

        offset_ticker += self.vertex_data.total_vertices * 8
        offset_ticker = align(offset_ticker, 32)

        ## -- BONE CHUNK --
        print("[>] Generating bone data...")

        self.offset_bone_parents = offset_ticker
        self.bone_parents = WMBBoneParents(arm_obj)

        offset_ticker += self.bone_count * 2
        offset_ticker = align(offset_ticker, 32)
        
        self.offset_bone_rel_positions = offset_ticker
        offset_ticker += self.bone_count * 12
        offset_ticker = align(offset_ticker, 32)
        self.offset_bone_abs_positions = offset_ticker

        self.bone_positions = WMBBonePosition(arm_obj)

        offset_ticker += self.bone_count * 12
        offset_ticker = align(offset_ticker, 32)

        self.offset_bone_index_translate_table = offset_ticker

        self.bone_index_translate_table = WMBBoneIndexTranslateTable(arm_obj)

        offset_ticker += self.bone_index_translate_table.size
        offset_ticker = align(offset_ticker, 32)

        self.bone_inverse_kinetic_table = WMBInverseKinetic(arm_obj)
        if (self.bone_inverse_kinetic_table.enabled):
            self.offset_bone_inverse_kinetic_table = offset_ticker
            offset_ticker += self.bone_inverse_kinetic_table.size
            offset_ticker = align(offset_ticker, 32)
        else:
            self.offset_bone_inverse_kinetic_table = 0

        self.bone_sym = WMBBoneSymmetries(arm_obj)
        if (self.bone_sym.enabled):
            self.offset_bone_sym = offset_ticker
            offset_ticker += self.bone_count * 2
            offset_ticker = align(offset_ticker, 32)
        else:
            self.offset_bone_sym = 0

        self.bone_flags = WMBBoneFlags(arm_obj)
        if (self.bone_flags.enabled):
            self.offset_bone_flags = offset_ticker
            offset_ticker += self.bone_count
            offset_ticker = align(offset_ticker, 32)
        else:
            self.offset_bone_flags = 0

        ## -- MATERIAL CHUNK --
        print("[>] Generating material data...")

        self.mat_blob = WMBMaterialBlob(arm_obj)
        self.mat_offset_offset = offset_ticker

        offset_ticker += 4 * self.mat_blob.material_count
        offset_ticker = align(offset_ticker, 32)

        self.mat_offset = offset_ticker

        offset_ticker += self.mat_blob.total_size
        offset_ticker = align(offset_ticker, 32)
        ## -- MESH CHUNK --
        print("[>] Generating mesh data...")
        self.mesh_blob = WMBMeshBlob(arm_obj)
        self.mesh_offset_offset = offset_ticker

        offset_ticker+=4*self.mesh_blob.mesh_count

        offset_ticker = align(offset_ticker, 32)
        self.mesh_offset = offset_ticker

        self.meshes = []
        self.mesh_offsets = []

        mesh_offset_ticker = 0
        
        for obj in arm_obj.children:
            if obj.type != 'MESH':
                continue
            name_parts = obj.name.split("-")
            if (len(name_parts) == 3):
                if (int(name_parts[2]) == 0):
                    mesh_dat = WMBMesh(arm_obj, obj, offset_ticker)
                    self.mesh_blob.offsets.append(mesh_offset_ticker)
                    self.mesh_offsets.append(mesh_offset_ticker)
                    self.meshes.append(mesh_dat)
                    mesh_offset_ticker+=mesh_dat.fetch_size()

        


def WMB0_Write_HDR(f, generated_data : WMBDataGenerator):
    f.write(b'WMB\x00')
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", generated_data.vtx_format))
    f.write(struct.pack("<i", generated_data.vertex_data.total_vertices))
    f.write(struct.pack("<b", generated_data.vertex_data.num_mapping))
    f.write(struct.pack("<b", generated_data.vertex_data.num_color))
    f.write(struct.pack("<h", 0))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", generated_data.offset_vertexes))
    f.write(struct.pack("<I", generated_data.offset_ex_vertexes))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 0))
    # Bone Chunk
    f.write(struct.pack("<I", generated_data.bone_count))
    f.write(struct.pack("<I", generated_data.offset_bone_parents))
    f.write(struct.pack("<I", generated_data.offset_bone_rel_positions))
    f.write(struct.pack("<I", generated_data.offset_bone_abs_positions))
    f.write(struct.pack("<I", generated_data.offset_bone_index_translate_table))
    # Material Chunk
    f.write(struct.pack("<I", generated_data.mat_blob.material_count))
    f.write(struct.pack("<I", generated_data.mat_offset_offset))
    f.write(struct.pack("<I", generated_data.mat_offset))
    # Mesh Chunk
    f.write(struct.pack("<I", generated_data.mesh_blob.mesh_count))
    f.write(struct.pack("<I", generated_data.mesh_offset_offset))
    f.write(struct.pack("<I", generated_data.mesh_offset))

    f.seek(100)
    f.write(struct.pack("<I", generated_data.offset_bone_inverse_kinetic_table))
    f.write(struct.pack("<I", generated_data.offset_bone_sym))
    f.write(struct.pack("<I", generated_data.offset_bone_flags))

def WMB0_Write_VertexData(f, generated_data : WMBDataGenerator):
    for data in generated_data.vertex_data.vertex_infos:
        uv_bytes = float_to_half_bytes(data[5][0]) + float_to_half_bytes(1 - data[5][1])
        f.write(struct.pack("<fff", *data[0]))
        f.write(uv_bytes)

        nx = int(round(data[1][0] * 127))
        ny = int(round(data[1][1] * 127))
        nz = int(round(data[1][2] * 127))

        nx = max(-127, min(127, nx))
        ny = max(-127, min(127, ny))
        nz = max(-127, min(127, nz))

        f.write(struct.pack('<4b', 0, nz, ny, nx)) # Normals
        f.write(struct.pack("<i", 0)) # Tangents
        f.write(struct.pack("<BBBB", *data[3])) # Bone Indexes
        f.write(struct.pack("<BBBB", *data[4])) # Bone Weights
    f.seek(generated_data.offset_ex_vertexes)
    for data in generated_data.vertex_data.exvertex_infos:
        f.write(struct.pack("<bbbb", *data[0]))
        f.write(struct.pack("<hh", *data[1]))

def WMB0_Write_BoneParents(f, generated_data : WMBDataGenerator):
    bone_map = generated_data.bone_parents.bone_map

    max_index = max(bone_map.keys())

    for bone_index in range(max_index + 1):
        parent_index = bone_map.get(bone_index, -1)
        f.write(struct.pack("<h", parent_index))

def WMB0_Write_Positions_Rel(f, generated_data : WMBDataGenerator):
    bone_map = generated_data.bone_positions.bone_rel_map
    for bone_index in sorted(bone_map.keys()):
        position = bone_map[bone_index]
        f.write(struct.pack("<f", position[0]))
        f.write(struct.pack("<f", position[2]))
        f.write(struct.pack("<f", -position[1]))

def WMB0_Write_Positions_Abs(f, generated_data : WMBDataGenerator):
    bone_map = generated_data.bone_positions.bone_pos_map
    for bone_index in sorted(bone_map.keys()):
        position = bone_map[bone_index]
        f.write(struct.pack("<f", position[0]))
        f.write(struct.pack("<f", position[2]))
        f.write(struct.pack("<f", -position[1]))


def WMB0_Write_BoneIndexTranslateTable(f, generated_data : WMBDataGenerator):
    for i in generated_data.bone_index_translate_table.data:
        f.write(struct.pack("<H", i))

def WMB0_Write_IK(f, generated_data : WMBDataGenerator):
    if (generated_data.bone_inverse_kinetic_table.enabled):
        f.write(struct.pack("<b", generated_data.bone_inverse_kinetic_table.count))
        f.write(struct.pack("<bbb", *generated_data.bone_inverse_kinetic_table.data))
        f.write(struct.pack("<i", generated_data.bone_inverse_kinetic_table.offset))
        for table in generated_data.bone_inverse_kinetic_table.structures:
            f.write(struct.pack("<" + ("b" * 16), *table))

def WMB0_Write_Sym(f, generated_data : WMBDataGenerator):
    if (generated_data.bone_sym.enabled):
        bone_map = generated_data.bone_sym.sym_map

        max_index = max(bone_map.keys())
        for bone_index in range(max_index + 1):
            sym_idx = bone_map.get(bone_index, -1)
            f.write(struct.pack("<h", sym_idx))

def WMB0_Write_Flags(f, generated_data : WMBDataGenerator):
    if (generated_data.bone_flags.enabled):
        bone_map = generated_data.bone_flags.flag_map

        max_index = max(bone_map.keys())
        for bone_index in range(max_index + 1):
            sym_idx = bone_map.get(bone_index, -1)
            f.write(struct.pack("<B", sym_idx))

def WMB0_Write_Mat_Offsets(f, generated_data : WMBDataGenerator):
    for ofst in generated_data.mat_blob.offsets:
        f.write(struct.pack("<I", ofst))

def WMB0_Write_Mat(f, generated_data : WMBDataGenerator):
    for mat in generated_data.mat_blob.materials:
        mat.write(f)

def WMB0_Write_Mesh_Offsets(f, generated_data : WMBDataGenerator):
    for ofst in generated_data.mesh_blob.offsets:
        f.write(struct.pack("<I", ofst))

def WMB0_Write_Mesh_Data(f, generated_data : WMBDataGenerator):
    for i, mesh in enumerate(generated_data.meshes):
        mesh_pos = generated_data.mesh_offset + generated_data.mesh_offsets[i]
        f.seek(mesh_pos)
        print(f"Writing mesh {mesh.name} @ {generated_data.mesh_offset + generated_data.mesh_offsets[i]}")

        f.write(struct.pack("<h", mesh.mesh_id))
        f.write(struct.pack("<h", mesh.batch_count))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<h", mesh.bounding_box_infos))
        f.write(struct.pack("<I", mesh.batch_offset_offset))
        f.write(struct.pack("<i", mesh.flags))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(mesh.name.ljust(32, '\x00').encode('utf-8'))
        f.write(struct.pack('<3ff3f3fff', *mesh.exdata))

        f.seek(mesh_pos + mesh.batch_offset_offset)
        print(f"|- Writing batch offsets {mesh_pos + mesh.batch_offset_offset}")
        for ofst in mesh.batch_offsets:
            f.write(struct.pack("<I", ofst))

        for i, batch in enumerate(mesh.batches):
            
            batch_offset = mesh_pos + mesh.batch_offset_offset + mesh.batch_offsets[i]
            print(f"   |- Writing batch @ {batch_offset}  | {len(batch.indices)} indice(s) @ {batch_offset + batch.indice_offset}")
            f.seek(batch_offset)
            f.write(struct.pack("<h", batch.batch_idx))
            f.write(struct.pack("<h", batch.id))
            f.write(struct.pack("<H", batch.flags))
            f.write(struct.pack("<h", batch.exmaterial_id))
            f.write(struct.pack("<B", batch.material_id))
            f.write(struct.pack("<B", batch.has_bone_refs))
            f.write(struct.pack("<b", 0))
            f.write(struct.pack("<b", 0))
            f.write(struct.pack("<I", batch.vertex_start))
            f.write(struct.pack("<I", batch.vertex_end))
            f.write(struct.pack("<I", batch.primitive_type))
            f.write(struct.pack("<I", batch.indice_offset))
            f.write(struct.pack("<I", len(batch.indices)))
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("<IIIIIII", 0, 0, 0, 0, 0, 0, 0))
            f.write(struct.pack("<I", len(batch.required_bones)))
            for i in batch.required_bones:
                f.write(struct.pack("<B", i))

            f.seek(batch_offset + batch.indice_offset)
            for indice in batch.indices:
                f.write(struct.pack("<H", indice))



def export(filepath, all_bone_refs=False):
    print("- BEGIN EXPORT -")
    f = open(filepath, "wb")
    print("[>] Preparing data...")

    generated_data = WMBDataGenerator() # Loosely based off of MGR2Blender
    f.seek(generated_data.header_offset)
    WMB0_Write_HDR(f, generated_data)
    f.seek(generated_data.offset_vertexes)
    WMB0_Write_VertexData(f, generated_data)
    f.seek(generated_data.offset_bone_parents)
    WMB0_Write_BoneParents(f, generated_data)
    f.seek(generated_data.offset_bone_rel_positions)
    WMB0_Write_Positions_Rel(f, generated_data)
    f.seek(generated_data.offset_bone_abs_positions)
    WMB0_Write_Positions_Abs(f, generated_data)
    f.seek(generated_data.offset_bone_index_translate_table)
    WMB0_Write_BoneIndexTranslateTable(f, generated_data)
    f.seek(generated_data.offset_bone_inverse_kinetic_table)
    WMB0_Write_IK(f, generated_data)
    f.seek(generated_data.offset_bone_sym)
    WMB0_Write_Sym(f, generated_data)
    f.seek(generated_data.offset_bone_flags)
    WMB0_Write_Flags(f, generated_data) # I should have called this bone flags but oh well
    f.seek(generated_data.mat_offset_offset)
    WMB0_Write_Mat_Offsets(f, generated_data)
    f.seek(generated_data.mat_offset)
    WMB0_Write_Mat(f, generated_data)
    f.seek(generated_data.mesh_offset_offset)
    WMB0_Write_Mesh_Offsets(f, generated_data)
    f.seek(generated_data.mesh_offset)
    WMB0_Write_Mesh_Data(f, generated_data)
    f.seek(align(f.tell(), 32))
    f.write(b"This WMB was brought to you by Gaming With Portals, Raq, and Skyth")

    f.close()
    return {'FINISHED'}
