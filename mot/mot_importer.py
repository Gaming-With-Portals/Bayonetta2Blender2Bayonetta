from .. import structwrapper
import bpy, math
from mathutils import Vector, radians

def getArmatureObject() -> bpy.types.Object: # MGR2Blender motUtils.py
	activeObj = bpy.context.active_object
	if activeObj is not None and activeObj.type == "ARMATURE":
		return activeObj
	allArmatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
	if len(allArmatures) == 0:
		return None
	wmbColl = bpy.data.collections.get("WMB")
	if wmbColl is None:
		return allArmatures[0]
	armaturesInWmbColl = [obj for obj in wmbColl.all_objects if obj.type == "ARMATURE"]
	if len(armaturesInWmbColl) == 0:
		return allArmatures[0]
	return armaturesInWmbColl[0]

FIXED_FLAG = "FIXED_T-POSE"
def fixTPose(armObj: bpy.types.Object):
    if FIXED_FLAG in armObj and armObj[FIXED_FLAG]:
        return
    print("Applying T-Pose")
    
    # change all the bone rotations (technically more accurate but less nice to work with)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in armObj.data.edit_bones:
        #length = math.dist(bone.head, bone.tail)
        bone.tail = Vector(bone.head) + Vector((0, 0.1, 0))
    
    # apply armature modifiers on all armature children
    child: bpy.types.Object
    for child in armObj.children:
        if child.type != "MESH":
            continue
        for mod in child.modifiers:
            if mod.type != "ARMATURE":
                continue
            bpy.context.view_layer.objects.active = child
            bpy.ops.object.modifier_apply(modifier=mod.name)
    
    # apply armature as rest pose
    bpy.context.view_layer.objects.active = armObj
    armObj.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode="OBJECT")
    
    # add armature modifier back
    for child in armObj.children:
        if child.type != "MESH":
            continue
        mod: bpy.types.ArmatureModifier
        mod = child.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = armObj

    armObj[FIXED_FLAG] = True

def objRotationWrapper(obj: bpy.types.Object):
	if obj.parent is not None \
		and obj.parent.name.startswith("RotationWrapper"):
		if abs(obj.parent.rotation_euler[0] - radians(90)) > 0.001:
			obj.parent.rotation_euler[0] = radians(90)
		if obj.rotation_euler[0] != 0:
			obj.rotation_euler[0] = 0
		return
	
	# make invisible parent with 90° rotation on x axis
	parentObj = bpy.data.objects.new("RotationWrapper", None)
	parentObj.hide_viewport = True
	parentObj.rotation_euler[0] = radians(90)
	obj.rotation_euler[0] = 0
	obj.users_collection[0].objects.link(parentObj)
	obj.parent = parentObj


def ImportMOT(filepath):
    realf = open(filepath, "rb")
    f = structwrapper.BinReader(realf)

    if (f.read(4) != b"mot\x00"):
        print("[!] Invalid MOT file")

    flag = f.read_u16()
    frameCount = f.read_u16()
    recordOffset = f.read_u32()
    recordNumber = f.read_u32()

    # Armature
    # Like all of this is stolen from MGR or Nier 2 Blendah
    arm_obj = getArmatureObject()
    if not arm_obj:
        print("[!] No model!")
        return
    
    for obj in [*arm_obj.pose.bones, arm_obj]:
        obj.location = (0, 0, 0)
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (0, 0, 0)
        obj.scale = (1, 1, 1)

    objRotationWrapper(arm_obj)




    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frameCount - 1
    bpy.context.scene.render.fps = 60
    