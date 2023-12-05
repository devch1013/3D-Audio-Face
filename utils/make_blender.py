import bpy
bs_name = [
    "AvatarMesh_Neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp_L",
    "browInnerUp_R",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff_L",
    "cheekPuff_R",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]

def make_blendshape(file_name:str, folder_path:str="/home/ubuntu/3d_temp/face_module/LDT/Results"):
    

    for name in bs_name:
        bpy.ops.wm.ply_import(filepath=f"{folder_path}/{name}.ply")

    bpy.data.objects.remove(bpy.data.objects["Camera"])
    bpy.data.objects.remove(bpy.data.objects["Light"])
    bpy.data.objects.remove(bpy.data.objects["Cube"])
    bpy.ops.object.select_all(action='DESELECT')
        
    for o in bpy.data.objects:
        # Check for given object names
        if o.name != "AvatarMesh_Neutral":
            o.select_set(True)
            
    bpy.context.view_layer.objects.active = bpy.data.objects["AvatarMesh_Neutral"]

    bpy.ops.object.join_shapes()
    bpy.ops.object.delete()

    # Save the modified main PLY file
    bpy.ops.wm.save_as_mainfile(filepath=f"/home/ubuntu/3d_temp/data/result_blend/{file_name}.blend")