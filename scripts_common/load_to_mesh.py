import os
import logging
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import igl
except ImportError:
    igl = None

try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except Exception:
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

log = logging.getLogger("comfyui_common")


def _load_mesh_file(file_path: str):
    """
    自包含的 mesh 加载逻辑（与 UniRig 行为一致）。
    支持 OBJ / PLY / STL / OFF / GLB / GLTF 等 trimesh 支持的格式。
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.fbx':
        return None, (
            "FBX file format is not directly supported.\n"
            "Please convert your FBX to GLB/OBJ format first."
        )

    if trimesh is None:
        return None, "trimesh is not installed."

    try:
        loaded = trimesh.load(file_path, process=False, maintain_order=True)

        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        # 确保三角面
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0 and mesh.faces.shape[1] != 3:
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False,
                maintain_order=True
            )

        # 清理：移除重复面和退化面（不 merge_vertices，避免破坏 UV/材质）
        unique_mask = mesh.unique_faces()
        nondegenerate_mask = mesh.nondegenerate_faces()
        valid_mask = unique_mask & nondegenerate_mask
        if not valid_mask.all():
            mesh.update_faces(valid_mask)

        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = ext

        return mesh, ""

    except Exception as e:
        if igl is None:
            return None, f"trimesh failed: {str(e)}"

        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"

            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)

            unique_mask = mesh.unique_faces()
            nondegenerate_mask = mesh.nondegenerate_faces()
            valid_mask = unique_mask & nondegenerate_mask
            if not valid_mask.all():
                mesh.update_faces(valid_mask)

            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = ext
            return mesh, ""

        except Exception as e2:
            return None, f"Error loading mesh: {str(e)}; Fallback error: {str(e2)}"


class UniRigLoadMeshDynamic:
    """
    API 友好的 Mesh 加载节点。
    file_path 为 STRING 类型，可接受任意动态路径，不再受 Combo 列表限制。
    输出类型为 TRIMESH，与 UniRig 下游节点完全兼容。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": (["input", "output"], {"default": "input"}),
                "file_path": ("STRING", {
                    "default": "3d/realistic_male_character.glb",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "execute"
    CATEGORY = "Common/3D"

    def execute(self, source_folder, file_path):
        if not file_path or not file_path.strip():
            raise ValueError("File path cannot be empty")

        full_path = None
        searched = []

        # 1. 按 source_folder 解析候选路径
        if source_folder == "input" and COMFYUI_INPUT_FOLDER is not None:
            candidates = [
                os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path),
                os.path.join(COMFYUI_INPUT_FOLDER, file_path),
            ]
        elif source_folder == "output" and COMFYUI_OUTPUT_FOLDER is not None:
            candidates = [os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)]
        else:
            candidates = []

        for path in candidates:
            searched.append(path)
            if os.path.exists(path):
                full_path = path
                break

        # 2. 回退到绝对路径
        if full_path is None:
            searched.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path

        if full_path is None:
            lines = "\n".join([f"  - {p}" for p in searched])
            raise ValueError(f"File not found: '{file_path}'\nSearched in:\n{lines}")

        # 3. 加载 mesh
        loaded_mesh, error = _load_mesh_file(full_path)
        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        log.info(
            "UniRigLoadMeshDynamic loaded: %s vertices, %s faces from %s",
            len(loaded_mesh.vertices), len(loaded_mesh.faces), full_path
        )
        return (loaded_mesh,)


NODE_CLASS_MAPPINGS = {
    "UniRigLoadMeshDynamic": UniRigLoadMeshDynamic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadMeshDynamic": "Common: Load Mesh (Dynamic Path)",
}