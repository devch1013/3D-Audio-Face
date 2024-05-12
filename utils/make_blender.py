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

def make_blendshape(file_name:str, meme:str = "ply", folder_path:str="face_module/LDT/Results"):
    
    
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    if meme == "ply":
        for name in bs_name:
            bpy.ops.wm.ply_import(filepath=f"{folder_path}/{name}.ply")
    elif meme == "obj":
        for name in bs_name:
            bpy.ops.wm.obj_import(filepath=f"{folder_path}/objs/{name}.obj")
    else:
        raise Exception("Choose valid meme of object")

    
    bpy.ops.object.select_all(action='DESELECT')
        
    for o in bpy.data.objects:
        # Check for given object names
        if o.name != "AvatarMesh_Neutral":
            o.select_set(True)
            
    bpy.context.view_layer.objects.active = bpy.data.objects["AvatarMesh_Neutral"]

    bpy.ops.object.join_shapes()
    bpy.ops.object.delete()
    
    for o in bpy.data.objects:
        # Check for given object names
        if o.name == "AvatarMesh_Neutral":
            o.select_set(True)
            o.data.use_auto_smooth = 0
        # ob.data.auto_smooth_angle = math.radians(40)  # 40 degrees as radians

    bpy.ops.object.shade_smooth()

    # Save the modified main PLY file
    # bpy.ops.wm.save_as_mainfile(filepath=f"data/result_blend/{file_name}.blend")
    bpy.ops.export_scene.fbx(filepath=f"data/result_fbx/{file_name}.fbx")
    
    
if __name__ == "__main__":
    make_blendshape(file_name = "test")