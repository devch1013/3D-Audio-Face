import os
import cv2
import open3d as o3d
from moviepy.editor import ImageSequenceClip
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import pymeshlab
import shutil

class HRN():
    def __init__(self, output_dir):
        self.hrn = pipeline(Tasks.face_reconstruction, model='damo/cv_resnet50_face-reconstruction', model_revision='v2.0.0-HRN')
        self.landmarks = [
            28112, 28788, 29178, 29383, 29550, 30289, 30455, 
            30663, 31057, 31717, 8162, 8178, 8188, 8193, 6516, 
            7244, 8205, 9164, 9884, 2216, 3887, 4921, 5829, 4802, 
            3641, 10456, 11354, 12384, 14067, 12654, 11493, 5523, 
            6026, 7496, 8216, 8936, 10396, 10796, 9556, 8837, 8237, 
            7637, 6916, 5910, 7385, 8224, 9065, 10538, 8830, 8230, 7630
        ]
        self.preserved_face = [
            55437, 56793, 57577, 57991, 58325, 59803, 60135, 60547, 61331, 62645, 
            15913, 15945, 15965, 15974, 12673, 14109, 15998, 17885, 19299, 4140, 
            7457, 9509, 11311, 9272, 6969, 20415, 22195, 24239, 27578, 24775, 22470, 
            10705, 11702, 14605, 16021, 17437, 20300, 21089, 18655, 17240, 16061, 
            14880, 13462, 11472, 14387, 16037, 17691, 20577, 17227, 16046, 14867
        ] 
        self.output_dir = output_dir
        self.downsampled_dir = os.path.join(output_dir, 'downsampled')
        self.query_vertices = None
        self.uv = None
    
    def save(self, result):
        os.makedirs(self.output_dir, exist_ok=True)
        mesh = result[OutputKeys.OUTPUT]['mesh']
        texture_map = result[OutputKeys.OUTPUT_IMG]
        mesh['texture_map'] = texture_map
        write_obj(os.path.join(self.output_dir, 'hrn_mesh_mid.obj'), mesh)

    def downsample(self, original_mesh_dir, face_name, targetfacenum=10000):
        condselect = ""
        for idx, face in enumerate(self.preserved_face):
            if (idx == len(self.preserved_face)-1) :
                condselect = condselect + f"(fi == {face})"
            else:
                condselect = condselect + f"(fi == {face}) || "
        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(original_mesh_dir, 'hrn_mesh_mid.obj'))
        ms.set_texture_per_mesh(textname=os.path.join(original_mesh_dir, 'hrn_mesh_mid.png'))
        ms.compute_selection_by_condition_per_face(condselect=condselect)
        ms.apply_selection_inverse(invfaces=True)
        ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=targetfacenum, preserveboundary=True, planarquadric=True, selected=True)
        
        os.makedirs(self.downsampled_dir, exist_ok=True)
        downsampled_output_path = os.path.join(self.downsampled_dir, 'hrn_mesh_mid.obj')
        ms.save_current_mesh(save_textures=True,file_name=downsampled_output_path)

        vertex_ids = []
        downsampled_mesh = o3d.io.read_triangle_mesh(downsampled_output_path)
        
        kdtree = o3d.geometry.KDTreeFlann(downsampled_mesh)
        for query_vertex in self.query_vertices:
            _, vertex_id, _ = kdtree.search_knn_vector_3d(query_vertex, 1)
            vertex_ids.append(vertex_id[0])
        
        landmark_txt_file = f"./face_module/LDT/data/landmarks/landmarks_{face_name}.txt"
        with open(landmark_txt_file, "w") as file:
            file.write(" ".join(map(str, vertex_ids)))
    
    def make_textures(self):
        folder_path = './face_module/LDT/Results'
        file_list = os.listdir(folder_path)
        os.makedirs(os.path.join(folder_path, 'objs'), exist_ok=True)
        exclude = ["iteration_1.ply", "iteration_2.ply", "iteration_3.ply", "iteration_4.ply", "Rescaled.ply", "Splitted.ply"]
        for file_name in file_list:
            if (file_name.endswith('.ply') and (file_name not in exclude)):
                file_path = os.path.join(folder_path, file_name)
                mesh_ply = o3d.io.read_triangle_mesh(file_path)
                mesh_ply.triangle_uvs = self.uv
                fn = file_name.split('.')[0]
                output_obj_path = os.path.join(folder_path, f'objs/{fn}.obj')
                o3d.io.write_triangle_mesh(output_obj_path, mesh_ply, write_triangle_uvs=True)

                with open(output_obj_path, 'r') as file:
                    lines = file.readlines()
                for i, line in enumerate(lines):
                    if line.startswith('mtllib'):
                        lines[i] = 'mtllib ./hrn_mesh_mid.mtl\n'
                        break
                with open(output_obj_path, 'w') as file:
                    file.writelines(lines)
        
        file_list = os.listdir(os.path.join(folder_path, 'objs'))
        for file_name in file_list:
            if file_name.endswith('.mtl'):
                os.remove(os.path.join(folder_path, 'objs', file_name))
        
        shutil.copy(os.path.join(self.downsampled_dir, 'hrn_mesh_mid.obj.mtl'), os.path.join(folder_path, 'objs', 'hrn_mesh_mid.mtl')) 
        shutil.copy(os.path.join(self.downsampled_dir, 'hrn_mesh_mid.png'), os.path.join(folder_path, 'objs', 'hrn_mesh_mid.png')) 
    
    def __call__(self, face_name, image_path):
        hrn_output = self.hrn(image_path)
        self.save(hrn_output)
        original_mesh = o3d.io.read_triangle_mesh(os.path.join(self.output_dir, 'hrn_mesh_mid.obj'))
        self.query_vertices = np.asarray(original_mesh.vertices)[self.landmarks]
        self.downsample(self.output_dir, face_name, targetfacenum=10000)
        mesh_obj = o3d.io.read_triangle_mesh(os.path.join(self.output_dir, 'downsampled', 'hrn_mesh_mid.obj'))
        self.uv = mesh_obj.triangle_uvs
        o3d.io.write_triangle_mesh(f'./face_module/LDT/Inputs/{face_name}.ply', mesh_obj, write_triangle_uvs=True, write_ascii=True)

