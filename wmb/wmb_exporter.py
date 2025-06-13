import bpy
import struct
from collections import defaultdict
import bmesh
from io import BufferedReader
class eMeshData:

    def __init__(self):
        self.name = ""
        self.batches = []
        self.mesh_id = 0
        self.exdata = []

    def write(self, f : BufferedReader):


        print(f"[>] Writing Mesh {self.name}")
        print(f"[ Batch Count: {len(self.batches)}")
        print(f"[ Mesh ID: {self.mesh_id}")
        f.write(struct.pack("<h", self.mesh_id))
        f.write(struct.pack("<h", len(self.batches)))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<h", 1))
        f.write(struct.pack("<I", 128))
        f.write(struct.pack("<i", -2147483648))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(self.name.ljust(32, '\x00').encode('utf-8'))
        f.write(struct.pack('<3ff3f3fff', *self.exdata))
        f.write(struct.pack("<i", -67372037))
        f.write(struct.pack("<i", -67372037))
        f.write(struct.pack("<i", -67372037))
        f.write(struct.pack("<i", -67372037))
        offset_ticker = (4*len(self.batches)) + 16
        for batch in self.batches:
            f.write(struct.pack("<i", offset_ticker))
            offset_ticker+=batch.fetch_size()
        f.write(struct.pack("<i", -67372037))
        f.write(struct.pack("<i", -67372037))
        f.write(struct.pack("<i", -67372037))
        f.write(struct.pack("<i", -67372037))

        for batch in self.batches:
            batch.write(f)

    def fetch_size(self):
        size = 0
        size+=112 # Header
        size+=16 # Padding?
        size+=4*len(self.batches) # Batch Offsets
        size+=16 # Padding 2?
        for batch in self.batches:
            size+=batch.fetch_size()


        return size

class eBatchData:


    def __init__(self):
        self.indices = []
        self.required_bones = []
        self.parent_mesh_id = 0
        self.material_id = 0
        self.vertex_start = 0
        self.vertex_end = 0
        self.flag = 0
        

    def fetch_size(self):
        size = 0
        size+=64 # Header
        size+=4 # Bone Ref Count
        size+=len(self.required_bones) # Bone Ref Table
        size+=len(self.indices) * 2 # Indice Data

        return size

    def write(self, f : BufferedReader):
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<h", self.parent_mesh_id))
        f.write(struct.pack("<H", 32769))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<b", self.material_id))
        f.write(struct.pack("<b", 1))
        f.write(struct.pack("<b", 0))
        f.write(struct.pack("<b", 0))
        f.write(struct.pack("<I", self.vertex_start))
        f.write(struct.pack("<I", self.vertex_end))
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<I", 68 + len(self.required_bones)))
        f.write(struct.pack("<I", len(self.indices)))
        f.write(struct.pack("<I", 0))

        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        f.write(struct.pack("<I", len(self.required_bones)))
       # for boneID in range(len(self.required_bones)):
        for boneID in range(len(self.required_bones)):
            f.write(struct.pack("<B", boneID))
        for indice in self.indices:
            f.write(struct.pack("<H", indice))

    
class eMaterial():
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

def create_wmb_header(f, sub_collection, data_pool):
    f.write(b"WMB\x00")
    f.write(struct.pack("<i", 0)) # Version probably idfk
    f.write(struct.pack("<i", sub_collection.collection["vertex_format"]))
    f.write(struct.pack("<i", len(data_pool["vertex"][0])))
    f.write(struct.pack("<b", (data_pool["vertex"][2])))
    f.write(struct.pack("<b", 1))
    f.write(struct.pack("<h", 0)) # Unknown value users when they meet padding users
    f.write(struct.pack("<i", 0)) # what the fuck
    f.write(struct.pack("<i", 128))
    f.write(struct.pack("<i", 128 + (len(data_pool["vertex"][0]) * data_pool["vtxsize"])))
    f.write(struct.pack("<i", 0)) # UNK G0
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", 0)) # UNK G3
    f.write(struct.pack("<i", data_pool["bonecount"]))
    offset = 128 + 8 + (len(data_pool["vertex"][0]) * (data_pool["vtxsize"] + data_pool["vtxsizeex"]))
    f.write(struct.pack("<i", offset)) # Why does EXData have 8 bytes of nothing? I have no idea! Anyways, bone parent table
    offset += (data_pool["bonecount"]) * 2
    f.write(struct.pack("<i", offset)) # Bone Relative Position
    offset += data_pool["bonecount"] * 12
    f.write(struct.pack("<i", offset)) # Bone Positions   
    offset += data_pool["bonecount"] * 12
    f.write(struct.pack("<i", offset)) # Bone Index Translate Table
    offset += data_pool["bonetts"]
    f.write(struct.pack("<i", len(data_pool["materials"])))
    f.write(struct.pack("<i", offset)) # okay so technically this offset should be calculated with the other stupid ass tables in mind but fuck that 
    offset += len(data_pool["materials"]) * 4
    f.write(struct.pack("<i", offset)) # material offsets
    for mat in data_pool["materials"]:
        offset+=mat.fetch_size()
    f.write(struct.pack("<i", len(data_pool["meshdata"].keys())))
    f.write(struct.pack("<i", offset)) # mesh offsets offsets
    offset += len(data_pool["meshdata"].keys()) * 4
    f.write(struct.pack("<i", offset)) # mesh offsets

    f.write(struct.pack("<i", 86654)) # What the fuck? I'm hard coding this for now this is... huh?!
    f.write(struct.pack("<i", 271)) # I'm hard coding this as well but at least this makes some sense

    f.write(struct.pack("<i", 0)) # Disable Inverse Kinetic
    f.write(struct.pack("<i", 0)) # Disable Bone Symmetries
    f.write(struct.pack("<i", 0)) # Disable Bone Flags

    f.write(struct.pack("<i", 0)) # (extra) material and shader data 
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", 0))




def get_bone_index_translate_table_size(sub_collection):
    arm_obj = sub_collection.collection.objects[0]

    return arm_obj["translate_table_size"]




def get_bone_count(sub_collection):
    arm_obj = sub_collection.collection.objects[0]
    return len(arm_obj.data.bones)

def gen_material_data(sub_collection):
    unique_mats = set()
    arm_obj = sub_collection.collection.objects[0]
    materials = []
    for child in arm_obj.children: 
        if child.type != 'MESH':
            continue
        for slot in child.material_slots:
            if slot.material:
                unique_mats.add(slot.material)

    for mat in unique_mats:
        emat = eMaterial()
        emat.id = int(mat.name[-1])
        emat.size = int(mat["size"])
        emat.flag = int(mat["flags"])
        emat.type = int(mat["type"])
        emat.data = mat["data"]
        materials.append(emat)



    return materials


def get_vertex_weights(vertex, obj, max_weights=4):
    weight_data = []

    for g in vertex.groups:
        group_index = g.group
        group_index = int(obj.vertex_groups[group_index].name[4:])
        #group_name = obj.vertex_groups[group_index].name
        weight = g.weight
        weight_data.append((group_index, weight))  # or use bone index mapping

    # Sort by weight (important!)
    weight_data.sort(key=lambda x: x[1], reverse=True)

    # Limit to top N weights (usually 4)
    weight_data = weight_data[:max_weights]

    # Normalize weights (sum should be 1.0)
    total = sum(w for _, w in weight_data)
    if total == 0:
        weight_data = [(0, 0.0)] * max_weights  # fallback
        total = 1.0

    normalized = [(idx, w / total) for idx, w in weight_data]

    # Convert to byte weights
    bone_indices = [idx for idx, _ in normalized]
    bone_weights = [int(w * 255) for _, w in normalized]

    # Pad to 4 if needed
    while len(bone_indices) < max_weights:
        bone_indices.append(0)
        bone_weights.append(0)
    return bone_indices, bone_weights

def gen_vertex_pool(sub_collection):
    arm_obj = sub_collection.collection.objects[0]
    tangent_map = defaultdict(list)
    position_data = []
    uv_data = []
    normal_data = []
    uv_layer_count = 0
    color_count = 1 
    colors = []
    bone_data = []
    tangents = []

    depsgraph = bpy.context.evaluated_depsgraph_get()
    for child in arm_obj.children:
        if child.type != 'MESH':
            continue

        child["vertex_start"] = len(position_data)

        if len(child.data.uv_layers) > uv_layer_count:
            uv_layer_count = len(child.data.uv_layers)

        eval_obj = child.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        eval_mesh.calc_tangents(uvmap=eval_mesh.uv_layers.active.name)
        uv_loop_data = eval_mesh.uv_layers.active.data if eval_mesh.uv_layers.active else None

        uv_map = {}
        sorted_loops = sorted(eval_mesh.loops, key=lambda loop: loop.vertex_index)
        for loop in sorted_loops:
            if uv_loop_data:
                uv = uv_loop_data[loop.index].uv
                vert_index = loop.vertex_index
                if vert_index not in uv_map:
                    uv_map[vert_index] = (uv.x, 1-uv.y)

                loopTangent = loop.tangent.normalized()
                tx = int(max(0, min(254, loopTangent.x * 127.0 + 127.0)))
                ty = int(max(0, min(254, loopTangent.y * 127.0 + 127.0)))
                tz = int(max(0, min(254, loopTangent.z * 127.0 + 127.0)))
                sign = 0xff if loop.bitangent_sign == -1 else 0x00
                tangents.append([tx, ty, tz, sign])

        for vert in eval_mesh.vertices:

            bone_data.append(get_vertex_weights(vert, child, 4))

            pos = child.matrix_world @ vert.co
            position_data.append((pos.x, pos.z, -pos.y)) 

            # Normal (world space)
            normal = child.matrix_world.to_3x3() @ vert.normal
            normal.normalize()
            normal_data.append((normal.x, normal.z, -normal.y)) 

            uv = uv_map.get(vert.index, (0.0, 0.0))
            uv_data.append(uv)

        eval_obj.to_mesh_clear()
        child["vertex_end"] = len(position_data)
    return (position_data, uv_data, uv_layer_count, color_count, colors, normal_data, bone_data, tangents)

def gen_mesh_data(sub_collection, data_pool):
    arm_obj = sub_collection.collection.objects[0]

    print("[>] Generating mesh data... this could take a moment")
    mesh_data_by_index = {}
    # First pass
    for child in arm_obj.children:
        if child.type != 'MESH':
            continue
        name_parts = child.name.split("-")
        if (len(name_parts) >= 3):
            if (int(name_parts[2]) == 0):
                tmp_mesh_data = eMeshData()
                tmp_mesh_data.exdata = child["data"]
                tmp_mesh_data.name = name_parts[1]
                tmp_mesh_data.mesh_id = int(name_parts[0])
                if (len(name_parts[1]) > 31):
                    print(f"[X] Skipped mesh {name_parts[1]}, the name is longer than 32 characters! This will probably cause an export error")
                    continue
                mesh_data_by_index[int(name_parts[0])] = tmp_mesh_data



    for child in arm_obj.children:
        if child.type != 'MESH':
            continue
        name_parts = child.name.split("-")
        if (len(name_parts) >= 3):
            batch_data = eBatchData()
            batch_data.material_id = int(child.material_slots[0].name[-1])
            batch_data.parent_mesh_id = int(name_parts[0])
            batch_data.vertex_start = child["vertex_start"]
            batch_data.vertex_end = child["vertex_end"]
            batch_data.flag = child["batch_flags"]
            for group in child.vertex_groups:
                batch_data.required_bones.append(int(group.name[4:]))
            batch_data.required_bones.sort()
            

            mesh = child.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)  # Convert to tris
            bm.to_mesh(mesh)
            bm.free()
            mesh.calc_loop_triangles()
            batch_data.indices = []
            for tri in mesh.loop_triangles:
                batch_data.indices.extend([
                    tri.vertices[0] + batch_data.vertex_start,
                    tri.vertices[1] + batch_data.vertex_start,
                    tri.vertices[2] + batch_data.vertex_start,
                ])

            mesh_data_by_index[int(name_parts[0])].batches.append(batch_data)

    for k, v in mesh_data_by_index.items():
        print(f"[Mesh {k}] -> {len(v.batches)} batches")

    return mesh_data_by_index


def write_vertex_pool(f, sub_collection, data_pool):
    print("[>] Writing Vertex Pool Data (1/2)")
    vertexes = data_pool["vertex"]
    for vertexid in range(len(vertexes[0])):
        f.write(struct.pack("<fff", *vertexes[0][vertexid]))
        f.write(struct.pack("<ee", *vertexes[1][vertexid]))
        nx = int(round(-vertexes[5][vertexid][0] * 127))
        ny = int(round(-vertexes[5][vertexid][1] * 127))
        nz = int(round(-vertexes[5][vertexid][2] * 127))

        # Clamp to valid byte range
        nx = max(-127, min(127, nx))
        ny = max(-127, min(127, ny))
        nz = max(-127, min(127, nz))

        f.write(struct.pack('<4b', 0, nz, ny, nx))
        f.write(struct.pack("<4B", *vertexes[7][vertexid]))
        f.write(struct.pack("<BBBB", *vertexes[6][vertexid][0]))
        f.write(struct.pack("<BBBB", *vertexes[6][vertexid][1]))
    print("[>] Writing Vertex[EX] Pool Data (2/2)")
    for vertexid in range(len(vertexes[0])):
        # EXData
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", 0))

def get_vetex_structure_size(sub_collection):
    vertexFormat = sub_collection.collection["vertex_format"]
    if vertexFormat == 0x4000001F:
        return 48
    else:
        return 32

def write_bone_parent_tree(f, sub_collection):
    print("[>] Writing Bone Hierarchy Map")
    arm_obj = sub_collection.collection.objects[0]
    f.write(struct.pack("<" + str(len(arm_obj["bone_hierarchy"])) + "h", *arm_obj["bone_hierarchy"]))

def write_bone_relative_positions(f, sub_collection):
    print("[>] Writing Bone Relative Position Map")
    position_start = f.tell()
    arm_obj = sub_collection.collection.objects[0]
    bones = arm_obj.data.bones
    bones_sorted = sorted(bones, key=lambda b: int(b.name[4:]))
    for i, bone in enumerate(bones_sorted):
        bonex = bone.tail.x
        boney = bone.tail.z
        bonez = -bone.tail.y 
        bonepx = 0
        bonepy = 0
        bonepz = 0
        if bone.parent is not None:
            bonepx = bone.parent.tail.x
            bonepy = bone.parent.tail.z
            bonepz = -bone.parent.tail.y   
        f.write(struct.pack("<fff", bonex - bonepx, boney - bonepy, bonez - bonepz))

def write_bone_translate_table(f, sub_collection):
    print("[>] Writing Bone Index Translate Table")
    arm_obj = sub_collection.collection.objects[0]
    f.write(struct.pack("<" + str(len(arm_obj["translate_table_1"])) + "h", *arm_obj["translate_table_1"]))
    f.write(struct.pack("<" + str(len(arm_obj["translate_table_2"])) + "h", *arm_obj["translate_table_2"]))
    f.write(struct.pack("<" + str(len(arm_obj["translate_table_3"])) + "h", *arm_obj["translate_table_3"]))

def write_bone_absolute_positions(f, sub_collection):
    print("[>] Writing Bone Absolute Position Map")
    position_start = f.tell()
    
    arm_obj = sub_collection.collection.objects[0]
    bones = arm_obj.pose.bones
    bones_sorted = sorted(bones, key=lambda b: int(b.name[4:]))
    for i, bone in enumerate(bones_sorted):
        bone_tail_world = arm_obj.matrix_world @ bone.tail

        bonex = bone_tail_world.x
        boney = bone_tail_world.z
        bonez = -bone_tail_world.y

        f.write(struct.pack("<fff", bonex, boney, bonez))
    

def write_materials(f, sub_collection, data_pool):

    offset_ticker = 0
    for mat in data_pool["materials"]:
        f.write(struct.pack("<I", offset_ticker))
        offset_ticker+=mat.fetch_size()

    for mat in data_pool["materials"]:
        mat.write(f)



def write_meshes(f, sub_collection, data_pool):
    arm_obj = sub_collection.collection.objects[0]
    mesh_dict = data_pool["meshdata"]
    offset_ticker = 0
    for key in sorted(mesh_dict.keys()):
        mesh_data = mesh_dict[key]
        f.write(struct.pack("<I", offset_ticker))
        offset_ticker+=mesh_data.fetch_size()

    mesh_dict = data_pool["meshdata"]
    for key in sorted(mesh_dict.keys()):
        mesh_data = mesh_dict[key]
        mesh_data.write(f)




def export(filepath):
    print("- BEGIN EXPORT -")
    f = open(filepath, "wb")
    print("[>] Preparing data...")


    wmb_layer_colllection = bpy.context.view_layer.layer_collection.children['WMB']
    sub_collection = [x for x in wmb_layer_colllection.children if x.is_visible][0]

    data_pool = {}
    data_pool["vertex"] = gen_vertex_pool(sub_collection)
    data_pool["vtxsize"] = get_vetex_structure_size(sub_collection)
    data_pool["bonecount"] = get_bone_count(sub_collection)
    data_pool["vtxsizeex"] = 8 # TOOD: Implement
    data_pool["bonetts"] = get_bone_index_translate_table_size(sub_collection)
    data_pool["materials"] = gen_material_data(sub_collection)
    data_pool["meshdata"] = gen_mesh_data(sub_collection, data_pool) 

    create_wmb_header(f, sub_collection, data_pool)
    f.seek(128)
    write_vertex_pool(f, sub_collection, data_pool)
    write_bone_parent_tree(f, sub_collection)
    write_bone_relative_positions(f, sub_collection)
    write_bone_absolute_positions(f, sub_collection)
    write_bone_translate_table(f, sub_collection)
    write_materials(f, sub_collection, data_pool)
    write_meshes(f, sub_collection, data_pool)

    f.close()
    return {'FINISHED'}
