import os
import cv2
from moviepy.editor import ImageSequenceClip
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_reconstruction = pipeline(Tasks.face_reconstruction, model='damo/cv_resnet50_face-reconstruction', model_revision='v2.0.0-HRN')

def save_results(result, save_root):
    os.makedirs(save_root, exist_ok=True)

    # export obj and texture
    mesh = result[OutputKeys.OUTPUT]['mesh']
    texture_map = result[OutputKeys.OUTPUT_IMG]
    mesh['texture_map'] = texture_map
    write_obj(os.path.join(save_root, 'hrn_mesh_mid.obj'), mesh)

def decimate_mesh(base_dir, output_dir='./downsampled', target_face_num=10000, obj_file_path=None, texture_file_path=None):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    if obj_file_path is not None:
        ms.load_new_mesh(obj_file_path)
        output_path = output_dir + os.path.base_name(obj_file_path)
    ms.load_new_mesh(os.path.join(base_dir, 'hrn_mesh_mid.obj'))
    if texture_file_path is not None:
        ms.set_texture_per_mesh(textname=texture_file_path)
    ms.set_texture_per_mesh(textname=os.path.join(base_dir, 'hrn_mesh_mid.png'))
    ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=target_face_num, preserveboundary=True)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'hrn_mesh_mid.obj')
    ms.save_current_mesh(save_textures=True,file_name=output_path)

    print(f'Output written to {os.path.abspath(output_path)}')

result = face_reconstruction('/home/elicer/haerin.png')
save_results(result, './hrn_outputs')
decimate_mesh('./hrn_outputs', output_dir='./hrn_outputs/downsampled')